from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

model = onnx.load("./yk_emotion.onnx")
onnx.checker.check_model(model)

quantize_dynamic("./yk_emotion.onnx", "./yk_emotion_int8.onnx", weight_type=QuantType.QUInt8)
