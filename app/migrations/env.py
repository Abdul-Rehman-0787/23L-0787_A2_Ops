from alembic import context
from sqlalchemy import create_engine

config = context.config


def runMigrations():
    engine = create_engine(config.get_main_option("sqlalchemy.url"))
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


runMigrations()