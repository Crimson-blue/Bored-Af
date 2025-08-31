import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Union

Number = Union[int, float]

class Matrix:
    MAX_N = 5
    TOL = 1e-12

    def __init__(self, rows_data: List[List[Number]]):
        if not rows_data or not isinstance(rows_data, list) or not isinstance(rows_data[0], list):
            raise ValueError("Data must be a non-empty list of lists.")
        row_count, column_count = len(rows_data), len(rows_data[0])
        if row_count < 1 or column_count < 1:
            raise ValueError("Matrix must have at least 1 row and 1 column.")
        if row_count > self.MAX_N or column_count > self.MAX_N:
            raise ValueError(f"Matrix size must be at most {self.MAX_N}x{self.MAX_N}.")
        for row in rows_data:
            if len(row) != column_count:
                raise ValueError("All rows must have the same number of columns.")
        self._data = [[float(value) for value in row] for row in rows_data]
        self._rows, self._cols = row_count, column_count

    @property
    def shape(self) -> Tuple[int, int]: return self._rows, self._cols
    @property
    def rows(self) -> int: return self._rows
    @property
    def cols(self) -> int: return self._cols
    @property
    def is_square(self) -> bool: return self.rows == self.cols

    def tolist(self) -> List[List[float]]: return [row[:] for row in self._data]
    def copy(self) -> "Matrix": return Matrix(self.tolist())
    def __repr__(self) -> str: return f"Matrix({self.tolist()})"

    def __str__(self) -> str:
        formatted = [[f"{x:.6g}" for x in row] for row in self._data]
        col_widths = [max(len(formatted[i][j]) for i in range(self.rows)) for j in range(self.cols)]
        lines = []
        for row_index in range(self.rows):
            parts = [formatted[row_index][j].rjust(col_widths[j]) for j in range(self.cols)]
            lines.append("[ " + "  ".join(parts) + " ]")
        return "\n".join(lines)

    def __getitem__(self, row_index: int) -> List[float]: return self._data[row_index][:]

    @staticmethod
    def zeros(row_count: int, column_count: int) -> "Matrix":
        if row_count < 1 or column_count < 1 or row_count > Matrix.MAX_N or column_count > Matrix.MAX_N:
            raise ValueError(f"Size must be within 1..{Matrix.MAX_N}")
        return Matrix([[0.0] * column_count for _ in range(row_count)])

    @staticmethod
    def identity(size: int) -> "Matrix":
        if size < 1 or size > Matrix.MAX_N: raise ValueError(f"Size must be within 1..{Matrix.MAX_N}")
        return Matrix([[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)])

    def _check_same_shape(self, other: "Matrix"):
        if self.shape != other.shape: raise ValueError("Matrices must have the same shape.")

    def __add__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([[self._data[i][j] + other._data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([[self._data[i][j] - other._data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, scalar: Number) -> "Matrix":
        if not isinstance(scalar, (int, float)): raise TypeError("Use @ for matrix multiplication; * is for scalar.")
        factor = float(scalar)
        return Matrix([[self._data[i][j] * factor for j in range(self.cols)] for i in range(self.rows)])

    def __rmul__(self, scalar: Number) -> "Matrix": return self.__mul__(scalar)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows: raise ValueError("Incompatible shapes for matrix multiplication.")
            result = Matrix.zeros(self.rows, other.cols)
            for row_index in range(self.rows):
                for k in range(self.cols):
                    a_value = self._data[row_index][k]
                    if abs(a_value) < self.TOL: continue
                    for col_index in range(other.cols):
                        result._data[row_index][col_index] += a_value * other._data[k][col_index]
            return result
        if isinstance(other, (list, tuple)):
            if len(other) != self.cols: raise ValueError("Vector length must match matrix column count.")
            output_vector = [0.0] * self.rows
            for row_index in range(self.rows):
                total = 0.0
                for col_index in range(self.cols):
                    total += self._data[row_index][col_index] * float(other[col_index])
                output_vector[row_index] = total
            return output_vector
        raise TypeError("Unsupported operand for @.")

    def T(self) -> "Matrix":
        return Matrix([[self._data[i][j] for i in range(self.rows)] for j in range(self.cols)])

    def _check_row_index(self, row_index: int):
        if not (0 <= row_index < self.rows): raise IndexError("Row index out of range.")

    def swap_rows(self, first_row_index: int, second_row_index: int) -> "Matrix":
        self._check_row_index(first_row_index); self._check_row_index(second_row_index)
        if first_row_index != second_row_index:
            self._data[first_row_index], self._data[second_row_index] = self._data[second_row_index], self._data[first_row_index]
        return self

    def scale_row(self, row_index: int, factor: Number) -> "Matrix":
        self._check_row_index(row_index)
        factor = float(factor)
        self._data[row_index] = [factor * value for value in self._data[row_index]]
        return self

    def add_rows(self, source_row_index: int, destination_row_index: int, factor: Number = 1.0) -> "Matrix":
        self._check_row_index(source_row_index); self._check_row_index(destination_row_index)
        factor = float(factor)
        if abs(factor) < self.TOL: return self
        self._data[destination_row_index] = [
            self._data[destination_row_index][j] + factor * self._data[source_row_index][j]
            for j in range(self.cols)
        ]
        return self

    def ref(self, tolerance: float = None):
        if tolerance is None: tolerance = self.TOL
        work = self.copy(); pivot_column_indices = []; current_row_index = 0
        for column_index in range(work.cols):
            pivot_row_index = current_row_index; largest_abs_value = 0.0
            for candidate_row_index in range(current_row_index, work.rows):
                value = abs(work._data[candidate_row_index][column_index])
                if value > largest_abs_value:
                    largest_abs_value = value; pivot_row_index = candidate_row_index
            if largest_abs_value <= tolerance: continue
            if pivot_row_index != current_row_index:
                work._data[current_row_index], work._data[pivot_row_index] = work._data[pivot_row_index], work._data[current_row_index]
            pivot_value = work._data[current_row_index][column_index]
            for row_below in range(current_row_index + 1, work.rows):
                eliminate_factor = work._data[row_below][column_index] / pivot_value
                if abs(eliminate_factor) <= tolerance: continue
                for j in range(column_index, work.cols):
                    work._data[row_below][j] -= eliminate_factor * work._data[current_row_index][j]
                if abs(work._data[row_below][column_index]) < tolerance: work._data[row_below][column_index] = 0.0
            pivot_column_indices.append(column_index); current_row_index += 1
            if current_row_index == work.rows: break
        return work, pivot_column_indices

    def rref(self, tolerance: float = None):
        if tolerance is None: tolerance = self.TOL
        work = self.copy(); current_row_index = 0; pivot_column_indices = []
        for column_index in range(work.cols):
            pivot_row_index = current_row_index; largest_abs_value = 0.0
            for candidate_row_index in range(current_row_index, work.rows):
                value = abs(work._data[candidate_row_index][column_index])
                if value > largest_abs_value:
                    largest_abs_value = value; pivot_row_index = candidate_row_index
            if largest_abs_value <= tolerance: continue
            if pivot_row_index != current_row_index:
                work._data[current_row_index], work._data[pivot_row_index] = work._data[pivot_row_index], work._data[current_row_index]
            pivot_value = work._data[current_row_index][column_index]
            if abs(pivot_value) <= tolerance: continue
            inv_pivot = 1.0 / pivot_value
            work._data[current_row_index] = [x * inv_pivot for x in work._data[current_row_index]]
            for row_index in range(work.rows):
                if row_index == current_row_index: continue
                factor = work._data[row_index][column_index]
                if abs(factor) <= tolerance: continue
                work._data[row_index] = [work._data[row_index][j] - factor * work._data[current_row_index][j] for j in range(work.cols)]
                if abs(work._data[row_index][column_index]) < tolerance: work._data[row_index][column_index] = 0.0
            pivot_column_indices.append(column_index); current_row_index += 1
            if current_row_index == work.rows: break
        for i in range(work.rows):
            for j in range(work.cols):
                if abs(work._data[i][j]) < tolerance: work._data[i][j] = 0.0
        return work, pivot_column_indices

    def det(self) -> float:
        if not self.is_square: raise ValueError("Determinant is defined for square matrices only.")
        work = self.copy(); n = work.rows; sign = 1.0; product = 1.0
        for column_index in range(n):
            pivot_row_index = column_index; largest_abs_value = 0.0
            for candidate_row_index in range(column_index, n):
                value = abs(work._data[candidate_row_index][column_index])
                if value > largest_abs_value:
                    largest_abs_value = value; pivot_row_index = candidate_row_index
            if largest_abs_value <= self.TOL: return 0.0
            if pivot_row_index != column_index:
                work._data[column_index], work._data[pivot_row_index] = work._data[pivot_row_index], work._data[column_index]
                sign *= -1.0
            pivot_value = work._data[column_index][column_index]; product *= pivot_value
            for row_below in range(column_index + 1, n):
                eliminate_factor = work._data[row_below][column_index] / pivot_value
                if abs(eliminate_factor) <= self.TOL: continue
                for j in range(column_index, n):
                    work._data[row_below][j] -= eliminate_factor * work._data[column_index][j]
        return float(sign * product)

    def inverse(self) -> "Matrix":
        if not self.is_square: raise ValueError("Inverse is defined for square matrices only.")
        n = self.rows
        augmented = [self._data[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        current_row_index = 0
        for column_index in range(n):
            pivot_row_index = current_row_index; largest_abs_value = 0.0
            for candidate_row_index in range(current_row_index, n):
                value = abs(augmented[candidate_row_index][column_index])
                if value > largest_abs_value:
                    largest_abs_value = value; pivot_row_index = candidate_row_index
            if largest_abs_value <= self.TOL: raise ValueError("Matrix is singular; inverse does not exist.")
            if pivot_row_index != current_row_index:
                augmented[current_row_index], augmented[pivot_row_index] = augmented[pivot_row_index], augmented[current_row_index]
            pivot_value = augmented[current_row_index][column_index]
            if abs(pivot_value) <= self.TOL: raise ValueError("Matrix is singular; inverse does not exist.")
            inv_pivot = 1.0 / pivot_value
            augmented[current_row_index] = [x * inv_pivot for x in augmented[current_row_index]]
            for row_index in range(n):
                if row_index == current_row_index: continue
                factor = augmented[row_index][column_index]
                if abs(factor) <= self.TOL: continue
                augmented[row_index] = [augmented[row_index][j] - factor * augmented[current_row_index][j] for j in range(2 * n)]
            current_row_index += 1
        return Matrix([row[n:] for row in augmented])

    def rank(self) -> int:
        _, pivot_columns = self.rref(); return len(pivot_columns)

    def trace(self) -> float:
        if not self.is_square: raise ValueError("Trace is defined for square matrices only.")
        return sum(self._data[i][i] for i in range(self.rows))


class MatrixPanel(ttk.Frame):
    def __init__(self, master, panel_name: str):
        super().__init__(master, padding=6)
        self.panel_name = panel_name
        self.rows_data: List[List[float]] = []
        self.column_count_var = tk.IntVar(value=3)
        self.columns_locked = False

        ttk.Label(self, text=f"Matrix {self.panel_name}", font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(self, text="Columns (1–5):").grid(row=1, column=0, sticky="w", padx=(0,4))
        self.column_spinner = ttk.Spinbox(self, from_=1, to=5, textvariable=self.column_count_var, width=4, command=self.rebuild_row_inputs)
        self.column_spinner.grid(row=1, column=1, sticky="w")

        self.row_input_container = ttk.Frame(self)
        self.row_input_container.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4,2))
        self.row_entry_widgets: List[tk.Entry] = []
        self.rebuild_row_inputs()

        ttk.Button(self, text="Add Row", command=self.add_row).grid(row=3, column=0, pady=(2,4), sticky="w")
        ttk.Button(self, text="Remove Last Row", command=self.remove_last_row).grid(row=3, column=1, pady=(2,4), sticky="w")
        ttk.Button(self, text="Clear", command=self.clear_rows).grid(row=3, column=2, pady=(2,4), sticky="w")
        self.rows_count_label = ttk.Label(self, text="Rows: 0/5")
        self.rows_count_label.grid(row=3, column=3, sticky="e")

        self.matrix_display_frame = ttk.Frame(self, padding=4)
        self.matrix_display_frame.grid(row=4, column=0, columnspan=4, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)
        self.refresh_display()

    def rebuild_row_inputs(self):
        for widget in self.row_input_container.winfo_children(): widget.destroy()
        column_count = self.column_count_var.get()
        self.row_entry_widgets = []
        ttk.Label(self.row_input_container, text="Next row:").grid(row=0, column=0, padx=(0,6))
        for column_index in range(column_count):
            entry = ttk.Entry(self.row_input_container, width=6)
            entry.grid(row=0, column=1 + column_index, padx=2, pady=2)
            self.row_entry_widgets.append(entry)

    def add_row(self):
        if len(self.rows_data) >= 5:
            messagebox.showerror("Limit reached", "Maximum 5 rows."); return
        column_count = self.column_count_var.get()
        try:
            new_row = [float(self.row_entry_widgets[j].get()) for j in range(column_count)]
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values for the entire row."); return
        if not new_row or len(new_row) != column_count:
            messagebox.showerror("Invalid row", "Row must have exactly the selected number of columns."); return
        if self.rows_data and len(self.rows_data[0]) != column_count:
            messagebox.showerror("Size mismatch", "Number of columns cannot change after adding rows."); return
        self.rows_data.append(new_row)
        for entry in self.row_entry_widgets: entry.delete(0, tk.END)
        self.refresh_display()
        if not self.columns_locked:
            self.column_spinner.state(["disabled"]); self.columns_locked = True

    def remove_last_row(self):
        if not self.rows_data: return
        self.rows_data.pop()
        self.refresh_display()
        if not self.rows_data and self.columns_locked:
            self.column_spinner.state(["!disabled"]); self.columns_locked = False

    def clear_rows(self):
        self.rows_data.clear()
        self.refresh_display()
        if self.columns_locked:
            self.column_spinner.state(["!disabled"]); self.columns_locked = False

    def refresh_display(self):
        for widget in self.matrix_display_frame.winfo_children(): widget.destroy()
        for row_index, row in enumerate(self.rows_data):
            for col_index, value in enumerate(row):
                ttk.Label(self.matrix_display_frame, text=f"{value:.6g}", width=8, anchor="e").grid(row=row_index, column=col_index, padx=2, pady=1, sticky="e")
        self.rows_count_label.config(text=f"Rows: {len(self.rows_data)}/5")

    def get_matrix(self) -> Matrix:
        if not self.rows_data: raise ValueError(f"Matrix {self.panel_name} is empty.")
        return Matrix(self.rows_data)

    def set_matrix(self, matrix_value: Matrix):
        self.rows_data = matrix_value.tolist()
        self.refresh_display()
        if self.rows_data:
            self.column_count_var.set(len(self.rows_data[0])); self.column_spinner.state(["disabled"]); self.columns_locked = True
        else:
            self.column_spinner.state(["!disabled"]); self.columns_locked = False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Matrix Builder + Operations")

        top_bar = ttk.Frame(self, padding=8); top_bar.pack(side="top", fill="x")
        ttk.Label(top_bar, text="Number of matrices:").pack(side="left")
        self.matrix_count_var = tk.IntVar(value=2)
        ttk.Spinbox(top_bar, from_=1, to=3, textvariable=self.matrix_count_var, width=4, command=self.update_matrix_visibility).pack(side="left", padx=6)

        panels_container = ttk.Frame(self, padding=8); panels_container.pack(side="top", fill="both", expand=True)
        self.matrix_panels = []
        for name in ["A", "B", "C"]:
            panel = MatrixPanel(panels_container, name)
            panel.grid(row=0, column=len(self.matrix_panels), padx=6, sticky="nsew")
            panels_container.columnconfigure(len(self.matrix_panels), weight=1)
            self.matrix_panels.append(panel)

        notebook = ttk.Notebook(self); notebook.pack(side="top", fill="both", expand=False, padx=8, pady=8)

        single_ops_tab = ttk.Frame(notebook, padding=8); notebook.add(single_ops_tab, text="Single-matrix Ops")
        ttk.Label(single_ops_tab, text="Target:").grid(row=0, column=0, sticky="w")
        self.selected_single_matrix_var = tk.StringVar(value="A")
        self.selected_single_matrix_menu = ttk.OptionMenu(single_ops_tab, self.selected_single_matrix_var, "A", "A", "B", "C")
        self.selected_single_matrix_menu.grid(row=0, column=1, sticky="w", padx=6)

        single_buttons = [
            ("Transpose", lambda: self.run_single_matrix_operation("T")),
            ("REF",       lambda: self.run_single_matrix_operation("REF")),
            ("RREF",      lambda: self.run_single_matrix_operation("RREF")),
            ("Determinant", lambda: self.run_single_matrix_operation("DET")),
            ("Inverse",   lambda: self.run_single_matrix_operation("INV")),
            ("Rank",      lambda: self.run_single_matrix_operation("RANK")),
            ("Trace",     lambda: self.run_single_matrix_operation("TRACE")),
        ]
        for idx, (label_text, handler) in enumerate(single_buttons):
            ttk.Button(single_ops_tab, text=label_text, command=handler).grid(row=1 + idx // 4, column=idx % 4, padx=4, pady=4, sticky="w")

        ttk.Label(single_ops_tab, text="Scalar value:").grid(row=3, column=0, sticky="w", pady=(8,2))
        self.scalar_value_var = tk.StringVar(value="2")
        ttk.Entry(single_ops_tab, textvariable=self.scalar_value_var, width=8).grid(row=3, column=1, sticky="w", padx=(6,0))
        ttk.Button(single_ops_tab, text="Scalar × Matrix", command=self.run_scalar_multiply).grid(row=3, column=2, padx=6, sticky="w")

        two_ops_tab = ttk.Frame(notebook, padding=8); notebook.add(two_ops_tab, text="Two-matrix Ops")
        ttk.Label(two_ops_tab, text="Left:").grid(row=0, column=0, sticky="w")
        ttk.Label(two_ops_tab, text="Right:").grid(row=0, column=2, sticky="w")
        self.left_matrix_name_var = tk.StringVar(value="A"); self.right_matrix_name_var = tk.StringVar(value="B")
        self.left_option_menu = ttk.OptionMenu(two_ops_tab, self.left_matrix_name_var, "A", "A", "B", "C")
        self.right_option_menu = ttk.OptionMenu(two_ops_tab, self.right_matrix_name_var, "B", "A", "B", "C")
        self.left_option_menu.grid(row=0, column=1, padx=6, sticky="w")
        self.right_option_menu.grid(row=0, column=3, padx=6, sticky="w")
        ttk.Button(two_ops_tab, text="A + B", command=lambda: self.run_two_matrix_operation("+")).grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(two_ops_tab, text="A − B", command=lambda: self.run_two_matrix_operation("-")).grid(row=1, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(two_ops_tab, text="A × B", command=lambda: self.run_two_matrix_operation("*")).grid(row=1, column=2, padx=4, pady=4, sticky="w")

        row_ops_tab = ttk.Frame(notebook, padding=8); notebook.add(row_ops_tab, text="Row Operations")
        ttk.Label(row_ops_tab, text="Target:").grid(row=0, column=0, sticky="w")
        self.row_ops_target_name_var = tk.StringVar(value="A")
        self.row_ops_target_menu = ttk.OptionMenu(row_ops_tab, self.row_ops_target_name_var, "A", "A", "B", "C")
        self.row_ops_target_menu.grid(row=0, column=1, padx=6, sticky="w")

        ttk.Label(row_ops_tab, text="Swap rows i, j (1-based):").grid(row=1, column=0, sticky="w", pady=(8,2))
        self.swap_row_i_var = tk.IntVar(value=1); self.swap_row_j_var = tk.IntVar(value=2)
        ttk.Spinbox(row_ops_tab, from_=1, to=5, textvariable=self.swap_row_i_var, width=4).grid(row=1, column=1, sticky="w")
        ttk.Spinbox(row_ops_tab, from_=1, to=5, textvariable=self.swap_row_j_var, width=4).grid(row=1, column=2, sticky="w", padx=(6,0))
        ttk.Button(row_ops_tab, text="Swap", command=self.run_swap_rows).grid(row=1, column=3, padx=6, sticky="w")

        ttk.Label(row_ops_tab, text="Scale row i by factor:").grid(row=2, column=0, sticky="w", pady=(8,2))
        self.scale_row_index_var = tk.IntVar(value=1); self.scale_factor_var = tk.StringVar(value="2")
        ttk.Spinbox(row_ops_tab, from_=1, to=5, textvariable=self.scale_row_index_var, width=4).grid(row=2, column=1, sticky="w")
        ttk.Entry(row_ops_tab, textvariable=self.scale_factor_var, width=6).grid(row=2, column=2, sticky="w", padx=(6,0))
        ttk.Button(row_ops_tab, text="Scale", command=self.run_scale_row).grid(row=2, column=3, padx=6, sticky="w")

        ttk.Label(row_ops_tab, text="R_dest += factor * R_src (1-based):").grid(row=3, column=0, sticky="w", pady=(8,2))
        self.add_src_row_index_var = tk.IntVar(value=1); self.add_dest_row_index_var = tk.IntVar(value=2); self.add_row_factor_var = tk.StringVar(value="1")
        ttk.Spinbox(row_ops_tab, from_=1, to=5, textvariable=self.add_src_row_index_var, width=4).grid(row=3, column=1, sticky="w")
        ttk.Spinbox(row_ops_tab, from_=1, to=5, textvariable=self.add_dest_row_index_var, width=4).grid(row=3, column=2, sticky="w")
        ttk.Entry(row_ops_tab, textvariable=self.add_row_factor_var, width=6).grid(row=3, column=3, sticky="w", padx=(6,0))
        ttk.Button(row_ops_tab, text="Add", command=self.run_add_rows).grid(row=3, column=4, padx=6, sticky="w")

        tools_bar = ttk.Frame(self, padding=(8,0)); tools_bar.pack(side="top", fill="x")
        ttk.Button(tools_bar, text="Clear Result", command=lambda: self.display_result("")).pack(side="right")

        self.result_text = tk.Text(self, height=14, font=("TkFixedFont", 10))
        self.result_text.pack(side="bottom", fill="both", expand=True, padx=8, pady=(0,8))

        self.update_matrix_visibility()

    def get_panel_for_name(self, name: str) -> MatrixPanel:
        return self.matrix_panels[{"A":0,"B":1,"C":2}[name]]

    def update_matrix_visibility(self):
        show_count = self.matrix_count_var.get()
        for index, panel in enumerate(self.matrix_panels):
            (panel.grid if index < show_count else panel.grid_remove)()
        self.sync_menus()

    def sync_menus(self):
        show_count = self.matrix_count_var.get()
        available_names = ["A", "B", "C"][:show_count]
        for variable, menu_widget in [
            (self.selected_single_matrix_var, self.selected_single_matrix_menu),
            (self.left_matrix_name_var, self.left_option_menu),
            (self.right_matrix_name_var, self.right_option_menu),
            (self.row_ops_target_name_var, self.row_ops_target_menu),
        ]:
            menu = menu_widget["menu"]; menu.delete(0, "end")
            if variable.get() not in available_names: variable.set(available_names[0])
            for name in available_names: menu.add_command(label=name, command=tk._setit(variable, name))

    def get_matrix_or_show_error(self, name: str):
        try:
            return self.get_panel_for_name(name).get_matrix()
        except Exception as error:
            messagebox.showerror("Matrix error", str(error)); return None

    def display_result(self, content):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", str(content) if content is not None else "")
        self.result_text.configure(state="disabled")

    def run_single_matrix_operation(self, operation: str):
        matrix_name = self.selected_single_matrix_var.get()
        matrix_value = self.get_matrix_or_show_error(matrix_name)
        if matrix_value is None: return
        try:
            if operation == "T": self.display_result(matrix_value.T())
            elif operation == "REF":
                ref_matrix, pivots = matrix_value.ref(); self.display_result(f"{ref_matrix}\n\nPivot columns: {pivots}")
            elif operation == "RREF":
                rref_matrix, pivots = matrix_value.rref(); self.display_result(f"{rref_matrix}\n\nPivot columns: {pivots}")
            elif operation == "DET": self.display_result(matrix_value.det())
            elif operation == "INV": self.display_result(matrix_value.inverse())
            elif operation == "RANK": self.display_result(matrix_value.rank())
            elif operation == "TRACE": self.display_result(matrix_value.trace())
        except Exception as error:
            messagebox.showerror("Operation error", str(error))

    def run_two_matrix_operation(self, operation: str):
        left_name = self.left_matrix_name_var.get()
        right_name = self.right_matrix_name_var.get()
        left_matrix = self.get_matrix_or_show_error(left_name)
        right_matrix = self.get_matrix_or_show_error(right_name)
        if left_matrix is None or right_matrix is None: return
        try:
            if operation == "+": self.display_result(left_matrix + right_matrix)
            elif operation == "-": self.display_result(left_matrix - right_matrix)
            elif operation in ("*", "@"): self.display_result(left_matrix @ right_matrix)
        except Exception as error:
            messagebox.showerror("Operation error", str(error))

    def run_scalar_multiply(self):
        matrix_name = self.selected_single_matrix_var.get()
        matrix_value = self.get_matrix_or_show_error(matrix_name)
        if matrix_value is None: return
        try:
            scalar_value = float(self.scalar_value_var.get())
        except ValueError:
            messagebox.showerror("Operation error", "Scalar must be numeric."); return
        try:
            self.display_result(scalar_value * matrix_value)
        except Exception as error:
            messagebox.showerror("Operation error", str(error))

    def run_swap_rows(self):
        matrix_name = self.row_ops_target_name_var.get()
        panel = self.get_panel_for_name(matrix_name)
        matrix_value = self.get_matrix_or_show_error(matrix_name)
        if matrix_value is None: return
        first_row = self.swap_row_i_var.get() - 1
        second_row = self.swap_row_j_var.get() - 1
        try:
            matrix_value.swap_rows(first_row, second_row); panel.set_matrix(matrix_value)
            self.display_result(f"Swapped rows {first_row+1} and {second_row+1} on {matrix_name}:\n\n{matrix_value}")
        except Exception as error:
            messagebox.showerror("Row op error", str(error))

    def run_scale_row(self):
        matrix_name = self.row_ops_target_name_var.get()
        panel = self.get_panel_for_name(matrix_name)
        matrix_value = self.get_matrix_or_show_error(matrix_name)
        if matrix_value is None: return
        row_index = self.scale_row_index_var.get() - 1
        try:
            factor = float(self.scale_factor_var.get())
        except ValueError:
            messagebox.showerror("Row op error", "Scale factor must be numeric."); return
        try:
            matrix_value.scale_row(row_index, factor); panel.set_matrix(matrix_value)
            self.display_result(f"Scaled row {row_index+1} by {factor} on {matrix_name}:\n\n{matrix_value}")
        except Exception as error:
            messagebox.showerror("Row op error", str(error))

    def run_add_rows(self):
        matrix_name = self.row_ops_target_name_var.get()
        panel = self.get_panel_for_name(matrix_name)
        matrix_value = self.get_matrix_or_show_error(matrix_name)
        if matrix_value is None: return
        source_row = self.add_src_row_index_var.get() - 1
        destination_row = self.add_dest_row_index_var.get() - 1
        try:
            factor = float(self.add_row_factor_var.get())
        except ValueError:
            messagebox.showerror("Row op error", "Factor must be numeric."); return
        if source_row == destination_row:
            messagebox.showerror("Row op error", "Source and destination rows must be different."); return
        try:
            matrix_value.add_rows(source_row, destination_row, factor); panel.set_matrix(matrix_value)
            self.display_result(f"Applied R{destination_row+1} += {factor} * R{source_row+1} on {matrix_name}:\n\n{matrix_value}")
        except Exception as error:
            messagebox.showerror("Row op error", str(error))

if __name__ == "__main__":
    App().mainloop()