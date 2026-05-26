from dagster import ScheduleDefinition, define_asset_job

pipeline_completo_job = define_asset_job(
    name="actualizar_modelo_mensual_job", 
    selection="*"
)

horario_mensual = ScheduleDefinition(
    job=pipeline_completo_job,
    cron_schedule="0 3 24 * *", 
    name="horario_entrenamiento_mensual",
    execution_timezone="Europe/Madrid"
)