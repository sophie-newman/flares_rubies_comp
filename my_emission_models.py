"""A script defining the FLARES emission model with AGN Template emission."""

import numpy as np
from synthesizer.emission_models import (
    BlackHoleEmissionModel,
    EmissionModel,
    GalaxyEmissionModel,
    NebularEmission,
    ReprocessedEmission,
    StellarEmissionModel,
    TemplateEmission,
    TransmittedEmission,
)
from synthesizer.emission_models.transformers import PowerLaw
from synthesizer.grid import Template
from unyt import Hz, Myr, angstrom, erg, nm, s


class AGNTemplateEmission(BlackHoleEmissionModel):
    """
    The stellar emission model used for in FLARES.

    This model is a subclass of the StellarEmissionModel class and is used
    to generate the stellar emission for galaxies in FLARES.
    """

    def __init__(self, agn_template_file, grid):
        """
        Initialize the FLARESLOSEmission model.

        Args:
            agn_template_file (str): The path to the AGN template file.
            grid (Grid): The grid object to unify the template with.
        """
        # Load the AGN template
        agn_template = np.loadtxt(
            agn_template_file,
            usecols=(0, 1),
            skiprows=23,
        )

        # Create the Template
        temp = Template(
            lam=agn_template[:, 0] * 0.1 * nm,
            lnu=agn_template[:, 1] * erg / s / Hz,
            unify_with_grid=grid,
        )

        # Create the agn template emission model
        agn_intrinsic = TemplateEmission(
            temp,
            emitter="blackhole",
            label="AGN_intrinsic",
        )

        # Define the attenuated AGN model
        BlackHoleEmissionModel.__init__(
            self,
            label="AGN_attenuated",
            apply_dust_to=agn_intrinsic,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
        )


class FLARESLOSCombinedEmission(EmissionModel):
    """
    The stellar emission model used for in FLARES.

    This model is a subclass of the StellarEmissionModel class and is used
    to generate the stellar emission for galaxies in FLARES.
    """

    def __init__(self, grid, agn_template_file):
        """
        Initialize the FLARESLOSEmission model.

        Args:
            grid (Grid): The grid to use for the model.
        """
        # Define the nebular emission models
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
        )
        transmitted = StellarEmissionModel(
            grid=grid,
            label="transmitted",
            combine=[young_transmitted, old_transmitted],
        )

        # Define the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            transmitted=young_transmitted,
            nebular=nebular,
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_transmitted],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_reprocessed,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_transmitted,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
        )

        # Finaly, combine to get the emergent emission
        total_stellar = StellarEmissionModel(
            grid=grid,
            label="stellar_total",
            combine=[young_attenuated, old_attenuated],
            related_models=[
                nebular,
                transmitted,
                reprocessed,
                young_attenuated_nebular,
            ],
        )

        # Load the AGN template
        agn_template = np.loadtxt(
            agn_template_file,
            usecols=(0, 1),
            skiprows=23,
        )

        # Create the Template
        temp = Template(
            lam=agn_template[:, 0] * angstrom,
            lnu=agn_template[:, 1] * erg / s / Hz,
            unify_with_grid=grid,
        )

        # Create the agn template emission model
        agn_intrinsic = TemplateEmission(
            temp,
            emitter="blackhole",
            label="agn_intrinsic",
        )

        # Define the attenuated AGN model
        agn_attenuated = BlackHoleEmissionModel(
            label="agn_attenuated",
            apply_dust_to=agn_intrinsic,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
        )

        # Make the combined galaxy intrinsic
        gal_intrinsic = GalaxyEmissionModel(
            grid=grid,
            label="combined_intrinsic",
            combine=(agn_intrinsic, reprocessed),
        )

        # Make model with dust and nebular free AGN 
        gal_dust_nebular_free = GalaxyEmissionModel(
            grid=grid,
            label="total_dust_nebular_free",
            combine=(agn_intrinsic, transmitted),
            emitter="galaxy",
        )

        # Make the combined total
        EmissionModel.__init__(
            self,
            grid=grid,
            label="total",
            combine=(agn_attenuated, total_stellar),
            related_models=[gal_intrinsic, gal_dust_nebular_free],
            emitter="galaxy",
        )

        self.set_per_particle(True)