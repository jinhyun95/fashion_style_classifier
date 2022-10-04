from torchvision import transforms


class Augmentation:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, (0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, x):
        x = self.transform(x)

        return x


class ToTensor:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, x):
        x = self.transform(x)

        return x