import gradio as gr
import os
import json
import shutil
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from predictor import UnimolPredictor
import zipfile
import uuid  # For generating unique filenames

# Define the predictions directory relative to app.py
BASE_DIR = Path(__file__).parent
PREDICTIONS_DIR = BASE_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

def convert_smiles_to_sdf(smiles, output_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    writer = Chem.SDWriter(output_path)
    writer.write(mol)
    writer.close()

def prepare_binding_box(uploaded_json):
    if uploaded_json is not None:
        with open(uploaded_json.name, 'r') as f:
            binding_box = json.load(f)
    else:
        raise ValueError("Binding box JSON file is required.")
    # Save binding box to the predictions directory
    binding_box_path = PREDICTIONS_DIR / f"binding_box_{uuid.uuid4().hex}.json"
    with open(binding_box_path, 'w') as f:
        json.dump(binding_box, f)
    return str(binding_box_path)

def merge_pdbs(protein_pdb, ligand_sdf, output_pdb):
    """
    Merge protein PDB and ligand SDF into a single PDB file for visualization.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Load protein
    protein_mol = Chem.MolFromPDBFile(protein_pdb, removeHs=False)
    if protein_mol is None:
        raise ValueError("Failed to load protein PDB file.")

    # Load ligand
    ligand_mol = Chem.MolFromMolFile(ligand_sdf, removeHs=False)
    if ligand_mol is None:
        raise ValueError("Failed to load ligand SDF file.")

    # Combine molecules
    combined = Chem.CombineMols(protein_mol, ligand_mol)

    # Write to PDB
    Chem.MolToPDBFile(combined, output_pdb)

def dock_single(protein_file, ligand_file, ligand_smiles, ligand_format, binding_box_json, conf_size, steric_clash_fix):
    try:
        # Define unique identifiers for filenames
        unique_id = uuid.uuid4().hex
        protein_filename = f"protein_{unique_id}.pdb"
        output_ligand_name = f"predicted_ligand_{unique_id}"
        output_sdf_filename = f"{output_ligand_name}.sdf"
        protein_path = PREDICTIONS_DIR / protein_filename
        output_sdf_path = PREDICTIONS_DIR / output_sdf_filename

        # Save protein file
        with open(protein_file.name, 'rb') as f_in, open(protein_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Handle ligand input
        if ligand_format == "SDF":
            ligand_path = PREDICTIONS_DIR / f"ligand_{unique_id}.sdf"
            with open(ligand_file.name, 'rb') as f_in, open(ligand_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            # Convert SMILES to SDF
            ligand_path = PREDICTIONS_DIR / f"ligand_{unique_id}.sdf"
            try:
                convert_smiles_to_sdf(ligand_smiles, ligand_path)
            except Exception as e:
                raise gr.Error(f"Error converting SMILES to SDF: {str(e)}")

        # Prepare binding box
        try:
            binding_box_path = prepare_binding_box(binding_box_json)
        except Exception as e:
            raise gr.Error(f"Error with binding box JSON: {str(e)}")

        # Set output directory
        output_dir = PREDICTIONS_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Initialize predictor
        predictor = UnimolPredictor.build_predictors(
            model_dir='../model/unimol_docking_v2_240517.pt',  # **IMPORTANT**: Ensure this points to the directory, not the .pt file
            mode='single',
            nthreads=8,
            conf_size=conf_size,
            cluster=False,
            use_current_ligand_conf=False,
            steric_clash_fix=steric_clash_fix
        )

        # Perform prediction
        try:
            _, _, _, output_sdf = predictor.predict_sdf(
                input_protein=str(protein_path),
                input_ligand=str(ligand_path),
                input_docking_grid=binding_box_path,
                output_ligand_name=output_ligand_name,
                output_ligand_dir=str(output_dir)
            )
            print(f"Prediction completed. Output SDF: {output_sdf}")
        except Exception as e:
            raise gr.Error(f"Error during prediction: {str(e)}")

        # Ensure output_sdf is a string, not a list
        if isinstance(output_sdf, list):
            if len(output_sdf) == 0:
                raise gr.Error("No output SDF files were generated.")
            output_sdf = output_sdf[0]

        # Check if the output file exists
        if not os.path.exists(output_sdf):
            raise gr.Error(f"Output SDF file not found: {output_sdf}")

        # Return the predicted SDF file
        return str(output_sdf)  # Return as a string

    except gr.Error as e:
        raise e  # Re-raise Gradio-specific errors for proper handling
    except Exception as e:
        raise gr.Error(f"Unexpected error: {str(e)}")

def dock_batch(protein_file, smiles_file, binding_box_json, conf_size, steric_clash_fix):
    try:
        # Define unique identifier for batch
        unique_id = uuid.uuid4().hex
        protein_filename = f"protein_batch_{unique_id}.pdb"
        protein_path = PREDICTIONS_DIR / protein_filename

        # Save protein file
        with open(protein_file.name, 'rb') as f_in, open(protein_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        # Read SMILES strings
        with open(smiles_file.name, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]

        if not smiles_list:
            return "SMILES file is empty or invalid."

        # Convert SMILES to SDF files
        ligand_paths = []
        valid_output_ligands = []
        for idx, smi in enumerate(smiles_list):
            sdf_filename = f"ligand_{unique_id}_{idx}.sdf"
            sdf_path = PREDICTIONS_DIR / sdf_filename
            try:
                convert_smiles_to_sdf(smi, sdf_path)
                ligand_paths.append(str(sdf_path))
                valid_output_ligands.append(f"predicted_ligand_{unique_id}_{idx}")
            except Exception as e:
                print(f"Skipping invalid SMILES at line {idx+1}: {smi}")

        if not ligand_paths:
            return "No valid SMILES strings found."

        # Prepare binding box
        try:
            binding_box_path = prepare_binding_box(binding_box_json)
        except Exception as e:
            return f"Error with binding box JSON: {str(e)}"

        # Set output directory
        output_dir = PREDICTIONS_DIR
        os.makedirs(output_dir, exist_ok=True)

        # Initialize predictor
        predictor = UnimolPredictor.build_predictors(
            model_dir='../model/unimol_docking_v2_240517.pt',  # **IMPORTANT**: Update this path
            mode='batch_one2many',
            nthreads=8,
            conf_size=conf_size,
            cluster=False,
            use_current_ligand_conf=False,
            steric_clash_fix=steric_clash_fix
        )

        # Perform prediction
        try:
            _, _, _, output_sdfs = predictor.predict_sdf(
                input_protein=str(protein_path),
                input_ligand=ligand_paths,
                input_docking_grid=binding_box_path,
                output_ligand_name=valid_output_ligands,
                output_ligand_dir=str(output_dir)
            )
            print(f"Batch prediction completed. Output SDFs: {output_sdfs}")
        except Exception as e:
            return f"Error during prediction: {str(e)}"

        # Zip all predicted SDF files
        zip_filename = f"predicted_ligands_{unique_id}.zip"
        zip_path = PREDICTIONS_DIR / zip_filename
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for sdf in output_sdfs:
                zipf.write(sdf, arcname=os.path.basename(sdf))

        # Check if the ZIP file exists
        if not os.path.exists(zip_path):
            return f"Output ZIP file not found: {zip_path}"

        # Return the ZIP file path
        return str(zip_path)

    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Unimol Docking Gradio App")
    
    with gr.Tab("Single Docking"):
        with gr.Column():
            protein_single = gr.File(label="Upload Protein PDB", file_types=[".pdb"])
            ligand_format = gr.Radio(choices=["SDF", "SMILES"], label="Ligand Input Format", value="SDF")
            ligand_file = gr.File(label="Upload Ligand SDF", file_types=[".sdf"], visible=True)
            ligand_smiles = gr.Textbox(label="Enter Ligand SMILES", lines=2, placeholder="Enter a SMILES string", visible=False)
            
            # Toggle visibility based on ligand format
            def toggle_ligand_input(format):
                if format == "SDF":
                    return (gr.update(visible=True), gr.update(visible=False))
                else:
                    return (gr.update(visible=False), gr.update(visible=True))
            
            ligand_format.change(
                toggle_ligand_input, 
                inputs=ligand_format, 
                outputs=[ligand_file, ligand_smiles]
            )
            
            # Binding box inputs - only JSON upload
            binding_box_json_single = gr.File(label="Upload Binding Box JSON", file_types=[".json"])
            
            # Parameters
            conf_size_single = gr.Number(label="Conf Size", value=10, step=1)
            steric_clash_fix_single = gr.Checkbox(label="Steric Clash Fix", value=True)
            
            # Docking button and outputs
            dock_single_btn = gr.Button("Run Docking")
            output_single_sdf = gr.File(label="Download Predicted Ligand SDF")
            # Removed Molecule3D visualization
            
            dock_single_btn.click(
                fn=dock_single,
                inputs=[
                    protein_single,
                    ligand_file,
                    ligand_smiles,
                    ligand_format,
                    binding_box_json_single,
                    conf_size_single,
                    steric_clash_fix_single
                ],
                outputs=output_single_sdf  # Only return the SDF file
            )
    
    with gr.Tab("Batch Docking"):
        with gr.Column():
            protein_batch = gr.File(label="Upload Protein PDB", file_types=[".pdb"])
            smiles_file = gr.File(label="Upload SMILES TXT", file_types=[".txt"])
            binding_box_json_batch = gr.File(label="Upload Binding Box JSON", file_types=[".json"])
            
            # Parameters
            conf_size_batch = gr.Number(label="Conf Size", value=10, step=1)
            steric_clash_fix_batch = gr.Checkbox(label="Steric Clash Fix", value=True)
            
            # Docking button and output
            dock_batch_btn = gr.Button("Run Batch Docking")
            output_batch = gr.File(label="Download Predicted Ligands ZIP")
            
            dock_batch_btn.click(
                fn=dock_batch,
                inputs=[
                    protein_batch,
                    smiles_file,
                    binding_box_json_batch,
                    conf_size_batch,
                    steric_clash_fix_batch
                ],
                outputs=output_batch
            )

demo.launch()
