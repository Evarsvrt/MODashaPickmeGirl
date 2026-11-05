import tkinter as tk
from tkinter import ttk, messagebox
import re
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def rosenbrock_fn(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2 for i in range(len(x) - 1))


def rosenbrock_grad(x):
    n = len(x)
    grad = np.zeros(n, dtype=float)
    grad[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
    for i in range(1, n - 1):
        grad[i] = 200.0 * (x[i] - x[i - 1] ** 2) - 400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
    grad[n - 1] = 200.0 * (x[n - 1] - x[n - 2] ** 2)
    return grad


def parse_function(expr_str, x0):
    clean = expr_str.strip()
    clean_compact = clean.replace(" ", "").lower()
    is_rosen = (
        "розенброк" in clean_compact
        or "rosenbrock" in clean_compact
        or "100*(x2-x1**2)**2+(1-x1)**2" in clean_compact
        or "100*(x_{i+1}-x_i^2)^2" in clean_compact
    )

    if is_rosen:
        n = len(x0)
        if n < 2:
            raise ValueError("Для функції Розенброка розмірність простору має бути m >= 2!")
        return rosenbrock_fn, rosenbrock_grad, n

    expr = sp.sympify(clean.replace("^", "**"))

    def sort_key(s):
        match = re.search(r"\d+", s.name)
        return (int(match.group()) if match else 0, s.name)

    variables = sorted(list(expr.free_symbols), key=sort_key)
    f_num = sp.lambdify(variables, expr, "numpy")
    grad_nums = [sp.lambdify(variables, sp.diff(expr, var), "numpy") for var in variables]

    f = lambda x: float(f_num(*x))
    grad = lambda x: np.array([float(g(*x)) for g in grad_nums], dtype=float)
    return f, grad, len(variables)


def golden_section(phi, a, b, delta):
    r = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = b - r * (b - a)
    x2 = a + r * (b - a)
    f1 = phi(x1)
    f2 = phi(x2)

    while (b - a) > delta:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - r * (b - a)
            f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + r * (b - a)
            f2 = phi(x2)

    return (a + b) / 2.0


def steepest_descent(f, grad_f, x0, eps, delta, K):
    x = np.array(x0, dtype=float)
    k = 0
    trajectory = [x.copy()]

    while True:
        p_k = -grad_f(x)
        norm_p = float(np.linalg.norm(p_k))

        phi = lambda alpha: f(x + alpha * p_k)
        alpha_k = golden_section(phi, 0.0, 1.0, delta)

        x_next = x + alpha_k * p_k
        trajectory.append(x_next.copy())

        if norm_p < eps or k >= K:
            return x_next, f(x_next), norm_p, k, np.array(trajectory)

        x = x_next
        k += 1


class OptimizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лабораторна робота №3 — Метод найшвидшого спуску")
        self.geometry("980x640")

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_main = ttk.Frame(notebook)
        notebook.add(tab_main, text="Оптимізація")

        tab_about = ttk.Frame(notebook)
        notebook.add(tab_about, text="About")

        tk.Label(
            tab_about,
            text="студентка: Міщук Дарина\nгрупа: ПМП-41",
            font=("Arial", 22),
            justify=tk.LEFT
        ).pack(anchor=tk.NW, padx=30, pady=30)

        left_panel = tk.Frame(tab_main, padx=10, pady=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(tab_main, padx=15, pady=15)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=left_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tk.Label(right_panel, text="Параметри оптимізації", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        tk.Label(right_panel, text="Цільова функція f(x):").pack(anchor=tk.W)
        self.entry_f = tk.Entry(right_panel, width=32)
        self.entry_f.insert(0, "Розенброка (m-вимірна)")
        self.entry_f.pack(pady=(0, 8))

        tk.Label(right_panel, text="Початкова точка x0 (m чисел через кому):").pack(anchor=tk.W)
        self.entry_x0 = tk.Entry(right_panel, width=32)
        self.entry_x0.insert(0, "-1.2, 1.0")
        self.entry_x0.pack(pady=(0, 8))

        tk.Label(right_panel, text="Точність eps (eps > 0):").pack(anchor=tk.W)
        self.entry_eps = tk.Entry(right_panel, width=32)
        self.entry_eps.insert(0, "0.001")
        self.entry_eps.pack(pady=(0, 8))

        tk.Label(right_panel, text="Точність delta (delta > 0):").pack(anchor=tk.W)
        self.entry_delta = tk.Entry(right_panel, width=32)
        self.entry_delta.insert(0, "0.00001")
        self.entry_delta.pack(pady=(0, 8))

        tk.Label(right_panel, text="Макс. ітерацій K (K > 1000):").pack(anchor=tk.W)
        self.entry_K = tk.Entry(right_panel, width=32)
        self.entry_K.insert(0, "2000")
        self.entry_K.pack(pady=(0, 12))

        btn_calc = tk.Button(right_panel, text="Розрахувати", font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", command=self.run_algorithm)
        btn_calc.pack(fill=tk.X, pady=(0, 15), ipady=3)

        res_frame = tk.LabelFrame(right_panel, text="Результати розрахунку", font=("Arial", 10, "bold"), padx=10, pady=10)
        res_frame.pack(fill=tk.X)

        self.lbl_dim = tk.Label(res_frame, text="Розмірність m: —", font=("Arial", 9, "bold"), anchor=tk.W, justify=tk.LEFT)
        self.lbl_dim.pack(fill=tk.X, pady=2)

        self.lbl_x = tk.Label(res_frame, text="x_{k+1}: —", anchor=tk.W, justify=tk.LEFT)
        self.lbl_x.pack(fill=tk.X, pady=2)

        self.lbl_f = tk.Label(res_frame, text="f(x_{k+1}): —", anchor=tk.W, justify=tk.LEFT)
        self.lbl_f.pack(fill=tk.X, pady=2)

        self.lbl_p = tk.Label(res_frame, text="||p_k||: —", anchor=tk.W, justify=tk.LEFT)
        self.lbl_p.pack(fill=tk.X, pady=2)

        self.lbl_k = tk.Label(res_frame, text="Ітерацій k: —", font=("Arial", 9, "bold"), anchor=tk.W, justify=tk.LEFT)
        self.lbl_k.pack(fill=tk.X, pady=2)

        self.after(100, self.run_algorithm)

    def run_algorithm(self):
        try:
            eps = float(self.entry_eps.get())
            delta = float(self.entry_delta.get())
            K = int(self.entry_K.get())

            if eps <= 0 or delta <= 0:
                messagebox.showerror("Помилка", "eps та delta повинні бути більшими за 0!")
                return
            if K <= 1000:
                messagebox.showerror("Помилка", "K має бути більшим за 1000!")
                return

            x0 = [float(val) for val in self.entry_x0.get().replace(",", " ").split()]
            if len(x0) == 0:
                messagebox.showerror("Помилка", "Введіть початкову точку x0!")
                return

            f, grad, dim = parse_function(self.entry_f.get(), x0)

            if len(x0) != dim:
                messagebox.showerror("Помилка", f"Розмірність x0 ({len(x0)}) не відповідає кількості змінних функції ({dim})!")
                return

            x_last, f_last, norm_p, iters, traj = steepest_descent(f, grad, x0, eps, delta, K)

            formatted_x = ", ".join([f"{val:.4f}" for val in x_last])
            self.lbl_dim.config(text=f"Розмірність m: {dim}")
            self.lbl_x.config(text=f"x_{{k+1}}: [{formatted_x}]")
            self.lbl_f.config(text=f"f(x_{{k+1}}): {f_last:.6e}")
            self.lbl_p.config(text=f"||p_k||: {norm_p:.6e}")
            self.lbl_k.config(text=f"Ітерацій k: {iters}")

            self.ax.clear()
            if dim == 2:
                xs, ys = traj[:, 0], traj[:, 1]
                pad_x = max(1.0, (np.max(xs) - np.min(xs)) * 0.3)
                pad_y = max(1.0, (np.max(ys) - np.min(ys)) * 0.3)
                gx = np.linspace(np.min(xs) - pad_x, np.max(xs) + pad_x, 100)
                gy = np.linspace(np.min(ys) - pad_y, np.max(ys) + pad_y, 100)
                X, Y = np.meshgrid(gx, gy)
                Z = np.zeros_like(X)
                for i in range(100):
                    for j in range(100):
                        Z[i, j] = f([X[i, j], Y[i, j]])

                try:
                    levels = np.logspace(np.log10(max(1e-2, np.min(Z) + 1e-4)), np.log10(np.max(Z)), 25)
                    self.ax.contour(X, Y, Z, levels=levels, cmap="Blues", alpha=0.7)
                except Exception:
                    self.ax.contour(X, Y, Z, levels=25, cmap="Blues", alpha=0.7)

                self.ax.plot(xs, ys, color="blue", marker=".", markersize=3, linewidth=1.2, label="Траєкторія")
                self.ax.plot(x0[0], x0[1], marker="o", color="navy", markersize=7, label="x0")
                self.ax.plot(x_last[0], x_last[1], marker="*", color="red", markersize=12, label="Мінімум x*")
                self.ax.set_title("Лінії рівня та траєкторія найшвидшого спуску (m = 2)")
                self.ax.set_xlabel("x1")
                self.ax.set_ylabel("x2")
                self.ax.grid(True, linestyle="--", alpha=0.5)
                self.ax.legend(loc="upper right")

            elif dim == 1:
                xs = traj[:, 0]
                pad = max(1.0, (np.max(xs) - np.min(xs)) * 0.5)
                gx = np.linspace(np.min(xs) - pad, np.max(xs) + pad, 150)
                gy = [f([val]) for val in gx]
                self.ax.plot(gx, gy, color="blue", linewidth=1.5, label="f(x)")
                traj_f = [f([val]) for val in xs]
                self.ax.plot(xs, traj_f, color="dodgerblue", marker="o", markersize=5, linestyle="--", label="Кроки спуску")
                self.ax.plot(x_last[0], traj_f[-1], marker="*", color="red", markersize=12, label="Мінімум x*")
                self.ax.set_title("Графік функції та кроки пошуку (m = 1)")
                self.ax.set_xlabel("x")
                self.ax.set_ylabel("f(x)")
                self.ax.grid(True, linestyle="--", alpha=0.5)
                self.ax.legend(loc="upper right")

            else:
                f_values = [f(pt) for pt in traj]
                self.ax.plot(range(len(f_values)), f_values, color="blue", linewidth=1.5, label="f(x_k)")
                self.ax.set_title(f"Графік збіжності функції f(x_k) (розмірність m = {dim})")
                self.ax.set_xlabel("Номер ітерації k")
                self.ax.set_ylabel("f(x_k)")
                self.ax.grid(True, linestyle="--", alpha=0.5)
                self.ax.legend(loc="upper right")

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося виконати розрахунок:\n{str(e)}")


if __name__ == "__main__":
    app = OptimizationApp()
    app.mainloop()
