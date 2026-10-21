"""Time one training epoch of each model planned for EN4554 A2 (planning tool, not part of the report).

Usage (GPU machine or Colab T4):  python tools/bench_epoch_time.py --out bench.json
Times real AMP training steps (batch 128, SGD+momentum) on random tensors and scales them to the
planned epochs. Prints seconds per epoch and the planned total for every experiment.
"""
import argparse, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):                       # Task 2(c) (bn=False) and Task 3(a) (bn=True)
    def __init__(self, bn=False):
        super().__init__()
        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=not bn)]
            layers += [nn.BatchNorm2d(cout)] if bn else []
            return layers + [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(2048, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 10))

    def forward(self, x):
        return self.head(self.features(x))


class Block(nn.Module):                          # Task 3(b) basic block, option-A shortcut
    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.residual, self.pad = residual, cout - cin
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.stride = stride

    def forward(self, x):
        out = self.b2(self.c2(F.relu(self.b1(self.c1(x)))))
        if self.residual:
            sc = x[:, :, ::self.stride, ::self.stride]
            if self.pad:
                sc = F.pad(sc, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
            out = out + sc
        return F.relu(out)


class CifarNet(nn.Module):                       # depth 6n+2, widths 16/32/64
    def __init__(self, n, residual):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        blocks, cin = [], 16
        for i, w in enumerate((16, 32, 64)):
            for j in range(n):
                blocks.append(Block(cin, w, 2 if (i > 0 and j == 0) else 1, residual)); cin = w
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.blocks(self.stem(x)), 1).flatten(1))


def cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class Res(nn.Module):
    def __init__(self, c):
        super().__init__(); self.f = nn.Sequential(cbr(c, c), cbr(c, c))

    def forward(self, x):
        return x + self.f(x)


def resnet9(c=16):                               # Task 4 candidate (ResNet-9 layout, base width c)
    return nn.Sequential(cbr(3, c), cbr(c, 2 * c), nn.MaxPool2d(2), Res(2 * c),
                         cbr(2 * c, 4 * c), nn.MaxPool2d(2), cbr(4 * c, 8 * c), nn.MaxPool2d(2),
                         Res(8 * c), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(8 * c, 10))


def time_model(make, bs=128, warm=15, steps=60):
    torch.manual_seed(0)
    m = make().cuda()
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    scaler = torch.amp.GradScaler("cuda")
    X = torch.randn(4096, 3, 32, 32, device="cuda"); Y = torch.randint(0, 10, (4096,), device="cuda")

    def step():
        idx = torch.randint(0, len(X), (bs,), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(m(X[idx]), Y[idx])
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    m.train()
    for _ in range(warm):
        step()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize(); t_step = (time.perf_counter() - t0) / steps

    m.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, 5000, 1000):
            m(X[:1000])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(0, 5000, 1000):
            m(X[:1000])
        torch.cuda.synchronize(); t_eval = time.perf_counter() - t0
    return sum(p.numel() for p in m.parameters()), t_step, t_eval


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="bench.json"); a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    models = {"SmallCNN": lambda: SmallCNN(False), "SmallCNN-BN": lambda: SmallCNN(True),
              "Net-8": lambda: CifarNet(1, True), "Net-20": lambda: CifarNet(3, True),
              "Net-56": lambda: CifarNet(9, True), "Plain-56": lambda: CifarNet(9, False),
              "ResNet9-w16": lambda: resnet9(16)}
    res = {}
    for k, f in models.items():
        p, ts, te = time_model(f)
        res[k] = {"params": p, "step_ms": ts * 1e3, "eval5k_s": te, "epoch45k_s": 352 * ts + te}
        print(f"{k:12s} params {p:>9,d} | {ts*1e3:6.2f} ms/step | 45k epoch {352*ts + te:6.2f} s")
    e = lambda k: res[k]["epoch45k_s"]
    small_epoch = 40 * res["SmallCNN-BN"]["step_ms"] / 1e3 + 2 * res["SmallCNN-BN"]["eval5k_s"]
    plan = {"T2c SmallCNN 9 runs x 15 ep": 9 * 15 * e("SmallCNN"),
            "T3a SmallCNN-BN 7 cfg x 60 ep (5k), 1 seed": 7 * 60 * small_epoch,
            "T3b 6 nets x 30 ep": 2 * 30 * (e("Net-8") + e("Net-20") + e("Net-56")),
            "T4 ResNet9-w16 5 runs x 20 ep": 5 * 20 * e("ResNet9-w16")}
    for k, v in plan.items():
        print(f"{k:45s} {v/60:6.1f} min")
    print(f"{'TOTAL (1 seed in T3a)':45s} {sum(plan.values())/60:6.1f} min | "
          f"3 seeds in T3a adds {2*plan['T3a SmallCNN-BN 7 cfg x 60 ep (5k), 1 seed']/60:.1f} min")
    json.dump({"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "models": res,
               "plan_s": plan}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
