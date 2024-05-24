import os
import json
import yaml
import multiprocessing as mp
import sys
from easydict import EasyDict
from tqdm import tqdm
from Bio.PDB import PDBParser
from functools import partial

sys.path.append('./esm')
from esm.inverse_folding.util import load_structure, extract_coords_from_structure


def extract_str(dir, pdb):
    file = pdb
    pdb = os.path.join(dir, pdb)
    chain_ids = set()
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb)

    for model in structure:
        for chain in model:
            chain_id = chain.id.strip()
            chain_ids.add(chain_id)

    for chain in chain_id:
        structure = load_structure(pdb, chain_id)
        coords, seq = extract_coords_from_structure(structure)
        # break
    return {"coords": coords.tolist(), "seq": seq, "name": file}


def get_file_names(directory):
    file_names = []
    for entry in os.scandir(directory):
        print(entry.name)
        if entry.is_file() and entry.name.endswith('.pdb'):
            file_names.append(entry.name)
            print(entry.name)
    return file_names


def prepare_dataset(config):
    if not os.path.exists(config.pre_process.dataset_path):
        raise ValueError("!!!train dataset not found!!!")
        
    #import ipdb; ipdb.set_trace()
    pdbs = get_file_names(config.pre_process.dataset_path)

    # extract_str(config.pre_process.dataset_path, pdbs[0])
    
    pool = mp.Pool(config.pre_process.pool_workers)
    with open(config.dataset.path, 'w') as f:
        for structure in tqdm(pool.imap_unordered(partial(extract_str, config.pre_process.dataset_path), pdbs), total=len(pdbs)):
            if structure is not None:
                f.write(f"{json.dumps(structure)}\n")
    pool.close()

if __name__ == '__main__':
    with open('./eei.yml', 'r') as f:
        config =  EasyDict(yaml.safe_load(f))
    prepare_dataset(config)