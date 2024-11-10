python demo.py --mode single --conf-size 10 --cluster \
  --input-protein ../example_data/protein.pdb \
  --input-ligand ../example_data/ligand.sdf \
  --input-docking-grid ../example_data/docking_grid.json \
  --output-ligand-name ligand_predict \
  --output-ligand-dir predicted_sdf \
  --steric-clash-fix \
  --model-dir ../model/unimol_docking_v2_240517.pt
