import onnx

model = onnx.load("zj_chocolate.onnx")

print("Inputs: ")
for i in model.graph.input:
    print(i.name, i.type)

print("\nOutputs: ")
for o in model.graph.output:
    print(o.name, o.type)
