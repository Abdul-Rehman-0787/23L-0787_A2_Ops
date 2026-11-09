"""remember which passages Artist used for each answer"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("chat_messages",sa.Column("sources", sa.Text(), nullable=False, server_default=""))


def downgrade():
    op.drop_column("chat_messages", "sources")