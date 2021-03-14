from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

model = ResNet50((32,32,3))

print(model.summary())
