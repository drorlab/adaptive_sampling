"""
Contains scaling function generators of all kinds
"""
from vmd import atomsel
import numpy as np

def get_scaler(molid, steepness=0.5):
    """
    Returns a scaling function appropriate for the given system.
    The function will smoothly switch from the normal contact
    distance to the maximal possible z distance based on box size
    at the point corresponding to the z dimension of the lowest
    lipid.

    Note that the scaling function accepts inputs in units of
    mdtraj nanometers, not VMD angstroms!!!

    Args:
        molid (int): VMD molecule ID to obtain scaler for
        steepness (float): Steepness of the switching function,
            or the width over which switching will be applied.
    Returns:
        (function handle): Scaling function to pass to featurizer
    """
    # Divide both of these by 10 since I need nanometers here
    selz = atomsel("protein and same fragment as resname ACE NMA", molid=molid).get('z')
    min_z = (max(selz)+min(selz))/2./10.

    def scaler(ligand_com, raw_dists):
        if len(ligand_com) != len(raw_dists):
            raise ValueError("Array size mismatch in scaling function")

        scale_factor = 0.5*np.tanh(steepness*(ligand_com[:, 2]-min_z))+0.5

        for i in range(raw_dists.shape[1]):
            raw_dists[:, i] = scale_factor/raw_dists[:,i]
        return raw_dists

    return scaler
