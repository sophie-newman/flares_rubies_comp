"""A script defining the pure stellar emission model used for LRDs in FLARES."""

import numpy as np
from synthesizer.emission_models import (
    BlackHoleEmissionModel,
    EmissionModel,
    EscapedEmission,
    GalaxyEmissionModel,
    NebularEmission,
    ReprocessedEmission,
    StellarEmissionModel,
    TemplateEmission,
    TransmittedEmission,
)
from synthesizer.emission_models.attenuation.dust import PowerLaw
from synthesizer.grid import Template
from unyt import Hz, Myr, angstrom, erg, s


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
        agn_template = np.loadtxt(agn_template_file, usecols=(0, 1), skiprows=23)

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
    The combined emission model used for in FLARES.

    This model is a subclass of the GalaxyEmissionModel class and is used
    to generate the combined emission for galaxies in FLARES.
    """

    def __init__(self, agn_template_file, grid, fesc=0.0, fesc_ly_alpha=1.0):
        """
        Initialize the FLARESLOSEmission model.

        Args:
            grid (Grid): The grid to use for the model.
        """
        # Define the incident models
        young_incident = StellarEmissionModel(
            grid=grid,
            label="young_incident",
            extract="incident",
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_incident = StellarEmissionModel(
            grid=grid,
            label="old_incident",
            extract="incident",
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        incident = StellarEmissionModel(
            grid=grid,
            label="incident",
            combine=[young_incident, old_incident],
        )

        # Define the nebular emission models
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        nebular = StellarEmissionModel(
            grid=grid,
            label="nebular",
            combine=[young_nebular, old_nebular],
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            fesc=fesc,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            fesc=fesc,
            mask_attr="ages",
            mask_op=">=",
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
            fesc=fesc,
            transmitted=young_transmitted,
            nebular=young_nebular,
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            fesc=fesc,
            transmitted=old_transmitted,
            nebular=old_nebular,
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_reprocessed],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_nebular,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1.3),
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_reprocessed,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-0.7),
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )

        # If we have an escape fraction, we need to include the escaped
        # emission
        young_escaped = EscapedEmission(
            grid,
            fesc=fesc,
            label="young_escaped",
            mask_attr="ages",
            mask_op="<",
            mask_thresh=10 * Myr,
        )
        old_escaped = EscapedEmission(
            grid,
            fesc=fesc,
            label="old_escaped",
            mask_attr="ages",
            mask_op=">=",
            mask_thresh=10 * Myr,
        )
        escaped = StellarEmissionModel(
            grid=grid,
            label="escaped",
            combine=[young_escaped, old_escaped],
        )

        # Define the intrinsc emission (we have this since there is an escape
        # fraction)
        young_intrinsic = StellarEmissionModel(
            grid=grid,
            label="young_intrinsic",
            combine=[young_reprocessed, young_escaped],
        )
        old_intrinsic = StellarEmissionModel(
            grid=grid,
            label="old_intrinsic",
            combine=[old_reprocessed, old_escaped],
        )
        intrinsic = StellarEmissionModel(
            grid=grid,
            label="stellar_intrinsic",
            combine=[young_intrinsic, old_intrinsic],
        )

        # Define the attenuated
        attenuated = StellarEmissionModel(
            grid=grid,
            label="stellar_attenuated",
            combine=[young_attenuated, old_attenuated],
        )

        # Finaly, combine to get the emergent emission
        total_stellar = StellarEmissionModel(
            grid=grid,
            label="stellar_total",
            combine=[escaped, attenuated],
            related_models=[
                incident,
                nebular,
                transmitted,
                reprocessed,
                young_attenuated_nebular,
                intrinsic,
            ],
        )

        # Load the AGN template
        agn_template = np.loadtxt(agn_template_file, usecols=(0, 1), skiprows=23)

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
            combine=(agn_intrinsic, intrinsic),
        )

        # Make model with dust free AGN but dust attenuated stellar emission
        gal_dust_free_agn = GalaxyEmissionModel(
            grid=grid,
            label="total_dust_free_agn",
            combine=(agn_intrinsic, total_stellar),
            emitter="galaxy",
        )

        # Make the combined total
        EmissionModel.__init__(
            self,
            grid=grid,
            label="total",
            combine=(agn_attenuated, total_stellar),
            related_models=[gal_intrinsic, gal_dust_free_agn],
            emitter="galaxy",
        )

        self.set_per_particle(True)