import tensorflow as tf
import tf2onnx

print("Loading Keras HDF5 model...")
model = tf.keras.models.load_model('fer2013_mini_XCEPTION.hdf5', compile=False)

# Define the input specification matching the model's expected shape: (batch_size, 48, 48, 1)
spec = (tf.TensorSpec((None, 48, 48, 1), tf.float32, name="input_1"),)

print("Converting to ONNX format...")
output_path = "fer2013_mini_XCEPTION.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

print(f"Success! Model saved to {output_path}")
