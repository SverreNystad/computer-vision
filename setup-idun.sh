# Define the dataset source path
DATASET_PATH="/cluster/projects/vc/data/ad/open/Poles"
echo "Dataset path: $DATASET_PATH"

# Define destination directories
DEST_DIR="/cluster/work/$USER/computer-vision/data"

COMBINED_DIR="$DEST_DIR/combined"

# Create the destination directories if they do not exist
mkdir -p "$DEST_DIR"
mkdir -p "$COMBINED_DIR"

# Copy the lidar data directory
echo "Copying lidar data..."
cp -r "$DATASET_PATH/lidar" "$DEST_DIR"
mv "$DEST_DIR/lidar/combined_color" "$DEST_DIR/lidar/images"

# Copy the rgb data directory
echo "Copying rgb data..."
cp -r "$DATASET_PATH/rgb" "$DEST_DIR"

# Copy the entire Poles directory into the combined folder
echo "Copying full Poles directory into combined folder..."
cp -r "$DEST_DIR/lidar/." "$COMBINED_DIR"
cp -r "$DEST_DIR/rgb/." "$COMBINED_DIR"

echo "Data configuration completed successfully."

# Setup environment
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.6.0

python -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt

deactivate

module purge
echo "Done setting up environment"
