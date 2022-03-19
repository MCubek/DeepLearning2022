import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# definiranje operacije
def f(x, a, b):
    return a * x + b


def multiclass_confusion_matrix(y_true, y_pred, class_count):
    with torch.no_grad():
        y_pred = F.one_hot(y_pred, class_count)
        cm = torch.zeros([class_count] * 2, dtype=torch.int64, device=y_true.device)
        for c in range(class_count):
            cm[c, :] = y_pred[y_true == c, :].sum(0)
        return cm


class Affine(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.out_features = out_features
    self.linear = torch.nn.Linear(in_features, out_features, bias=False)
    self.bias = torch.nn.Parameter(torch.zeros(out_features))

  def forward(self, input):
    return self.linear(input) + self.bias



if __name__ == '__main__':
    # definiranje varijabli i izgradnja dinamičnog
    # računskog grafa s unaprijednim prolazom
    a = torch.tensor(5., requires_grad=True)
    b = torch.tensor(8., requires_grad=True)
    x = torch.tensor(2.)
    y = f(x, a, b)
    s = a ** 2

    # unatražni prolaz koji računa gradijent
    # po svim tenzorima zadanim s requires_grad=True
    y.backward()
    s.backward()  # gradijent se akumulira
    assert x.grad is None  # pytorch ne računa gradijente po x
    assert a.grad == x + 2 * a  # dy/da + ds/da
    assert b.grad == 1  # dy/db + ds/db

    # ispis rezultata
    print(f"y={y}, g_a={a.grad}, g_b={b.grad}")

    affine = Affine(3, 4)

    print(list(affine.named_parameters()))

    dataset = [(torch.randn(4, 4), torch.randint(5, size=())) for _ in range(25)]
    dataset = [(x.numpy(), y.numpy()) for x, y in dataset]
    loader = DataLoader(dataset, batch_size=8, shuffle=False,
                        num_workers=0, collate_fn=None, drop_last=False)
    for x, y in loader:
        print(x.shape, y.shape)