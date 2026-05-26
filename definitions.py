from dagster import Definitions, load_assets_from_modules
import assets
import schedules

# Cargamos de golpe todos los activos que encuentre dentro del script assets.py
all_assets = load_assets_from_modules([assets])

# Definimos el repositorio global oficial de Dagster
defs = Definitions(
    assets=all_assets,
    schedules=[schedules.horario_mensual]
)