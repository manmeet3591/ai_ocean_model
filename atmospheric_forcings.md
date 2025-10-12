# Atmospheric Forcings for AI Ocean Model Training

This document outlines the key **atmospheric variables** to include when
training an **AI-driven ocean model**. These forcings represent the
external drivers influencing the ocean's surface state (e.g.,
temperature, salinity, and currents).

------------------------------------------------------------------------

## 🌤️ Core Atmospheric Forcings (Must-Have)

  --------------------------------------------------------------------------------------------
  **Category**       **Variable      **Typical Symbol**          **Units**   **Why It
                     Name**                                                  Matters**
  ------------------ --------------- --------------------------- ----------- -----------------
  **Wind**           10 m zonal wind `u10`                       m/s         Drives zonal
                                                                             surface currents
                                                                             and upwelling.

                     10 m meridional `v10`                       m/s         Drives meridional
                     wind                                                    transport and
                                                                             Ekman divergence.

                     Wind stress     `tau_x`, `tau_y`            N/m²        Momentum flux;
                     (optional,                                              can be derived
                     derived)                                                from winds.

  **Heat Fluxes**    Shortwave       `sw_down`                   W/m²        Direct solar
                     radiation                                               heating of
                                                                             surface layers.

                     Longwave        `lw_down`                   W/m²        Net cooling
                     radiation                                               through IR
                     (downward)                                              radiation.

                     Latent heat     `q_latent`                  W/m²        Cooling due to
                     flux                                                    evaporation.

                     Sensible heat   `q_sensible`                W/m²        Heating/cooling
                     flux                                                    from air--sea
                                                                             temperature
                                                                             gradient.

                     Net heat flux   `q_net`                     W/m²        Combined total
                                                                             heat exchange
                                                                             with atmosphere.

  **Hydrological**   Precipitation   `precip`                    kg/m²/s or  Adds freshwater,
                     rate                                        mm/day      stabilizes
                                                                             surface layer.

                     Evaporation     `evap`                      kg/m²/s or  Removes
                     rate                                        mm/day      freshwater,
                                                                             increases
                                                                             salinity.

                     Net freshwater  `fw_flux = precip - evap`   kg/m²/s     Controls surface
                     flux                                                    salinity balance.

  **Air Properties** Air temperature `t2m`                       K           Determines
                     (2 m)                                                   sensible heat
                                                                             exchange.

                     Specific        `q2m`                       kg/kg       Governs latent
                     humidity (2 m)                                          heat flux and
                                                                             evaporation.

                     Air pressure    `sp`                        Pa          Affects sea level
                     (surface)                                               (inverse
                                                                             barometer
                                                                             effect).

                     Cloud fraction  `cloud_frac`                0--1        Modulates
                                                                             radiation
                                                                             balance.
  --------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 🌪️ Optional / High-Value Features for AI Models

  ------------------------------------------------------------------------
  **Category**            **Variable**           **Why Useful**
  ----------------------- ---------------------- -------------------------
  **Turbulent Flux        Wind stress curl,      Key for identifying
  Components**            divergence             upwelling zones.

  **Atmospheric           Air--sea temperature   Controls surface flux
  Stability**             difference             regimes.
                          (`t2m - sst`)          

  **Relative Humidity**   RH (%)                 Alternative to specific
                                                 humidity for moisture
                                                 forcing.

  **Radiative             Upward longwave        To close the
  Components**            (`lw_up`), reflected   top-of-atmosphere budget.
                          shortwave (`sw_up`)    

  **Rain Evaporation**    Rain rate /            Important in tropics for
                          evaporation ratio      salinity and
                                                 stratification.
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 🧠 Feature Engineering Notes for AI Training

-   **Time Resolution:** Use 3-hourly or 6-hourly data (e.g., ERA5) ---
    finer than daily if possible.\
-   **Spatial Resolution:** Regrid to your ocean model resolution (e.g.,
    0.25° or 1°).\
-   **Normalization:**
    -   Normalize or standardize per-variable (e.g., z-score or
        min--max).\
    -   Maintain **physical consistency** (e.g., same sign conventions
        as ocean output).\
-   **Temporal Lags:**\
    Include lagged atmospheric variables (e.g., previous 1--3 days) to
    capture delayed ocean response.

------------------------------------------------------------------------

## 📡 Recommended Data Sources

  -----------------------------------------------------------------------
  **Dataset**       **Variables Available**          **Advantages**
  ----------------- -------------------------------- --------------------
  **ERA5 (ECMWF)**  All above                        Best spatial (0.25°)
                                                     and temporal
                                                     (hourly) coverage.

  **JRA55-do**      Optimized ocean forcing set      Balanced and
                                                     bias-corrected for
                                                     ocean models.

  **MERRA-2**       Includes aerosols and cloud      Useful for coupled
                    fields                           atmosphere--ocean AI
                                                     systems.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🔑 Minimal Forcing Set

If you want a compact but effective atmospheric input vector:

    [u10, v10, sw_down, lw_down, t2m, q2m, precip, sp]

This set captures most of the variability driving upper-ocean heat and
momentum exchange.

------------------------------------------------------------------------

© 2025 AI Ocean Modeling Guide
