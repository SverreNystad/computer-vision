
# Define the dataset source path
DATASET_PATH="/datasets/tdt4265/ad/open/Poles"
echo "Dataset path: $DATASET_PATH"

# Define destination directories
DEST_DIR="/work/$USER/computer-vision/data"
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
cp -r "$DATASET_PATH" "$COMBINED_DIR"

echo "Data configuration completed successfully."