"""Constants used in Cloud processing."""

CLEAR_SKY                       = 0   # Clear sky
CLOUD_LIQUID                    = 1   # Cloud liquid droplets only
DRIZZLE_OR_RAIN                 = 2   # Drizzle or rain
DRIZZLE_OR_RAIN_LIQUID_DROPLETS = 3   # Drizzle or rain coexisting with cloud liquid droplets
ICE_PARTICLES                   = 4   # Ice particles
ICE_WITH_SUP_WATER              = 5   # Ice coexisting with supercooled liquid droplets
MELTING_ICE                     = 6   # Melting ice particles
MELTING_ICE_LIQUID_DROPLETS     = 7   # Melting ice particles coexisting with cloud liquid droplets
AERO_NO_CLOUD                   = 8   # Aerosol particles, no cloud or precipitation
INSECT_NO_CLOUD                 = 9   # Insects, no cloud or precipitation
AERO_WITH_INSECT_NO_CLOUD       = 10  # Aerosol coexisting with insects, no cloud or precipitation

CLASSIFICATION_TICK_LABELS = ['Clear  sky',
               'Droplets',
               'Drizzle or rain',
               'Drizzle & droplets',
               'Ice',
               'Ice & droplets',
               'Melting ice',
               'Melting & droplets',
               'Aerosol',
               'Insect',
               'Aerosol & insect',
               'No Data']

TARG_BET_HYDRO = [CLEAR_SKY, AERO_NO_CLOUD, INSECT_NO_CLOUD, AERO_WITH_INSECT_NO_CLOUD]