from typing import List, Tuple, Union

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox,
    QTextEdit, QVBoxLayout, QWidget
)

Number = Union[int, float]

class Matrix:
    Maximum = 5

    def __init__(self, rows_data: List[List[Number]]):
        if not rows_data or not isinstance(rows_data, list) or not isinstance(rows_data[0], list):
            raise ValueError("Data must be a non-empty list of lists.")
        row_count, column_count = len(rows_data), len(rows_data[0])
        if row_count < 1 or column_count < 1:
            raise ValueError("Matrix must have at least 1 row and 1 column.")
        if row_count > self.Maximum or column_count > self.Maximum:
            raise ValueError(f"Matrix size must be at most {self.Maximum}x{self.Maximum}.")
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

    def tolist(self) -> List[List[float]]: return [row[:] for row in self._data]
    def __repr__(self) -> str: return f"Matrix({self.tolist()})"

    def __str__(self) -> str:
        formatted = [[f"{x:.6g}" for x in row] for row in self._data]
        col_widths = [max(len(formatted[i][j]) for i in range(self.rows)) for j in range(self.cols)]
        lines = []
        for row_index in range(self.rows):
            parts = [formatted[row_index][j].rjust(col_widths[j]) for j in range(self.cols)]
            lines.append("[ " + "  ".join(parts) + " ]")
        return "\n".join(lines)

    def _check_same_shape(self, other: "Matrix"):
        if self.shape != other.shape:
            raise ValueError(f"Matrices must have the same shape: {self.shape} vs {other.shape}")

    def __add__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([[self._data[i][j] + other._data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __sub__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other)
        return Matrix([[self._data[i][j] - other._data[i][j] for j in range(self.cols)] for i in range(self.rows)])

    def __mul__(self, other: Union["Matrix", Number]) -> "Matrix":
        if isinstance(other, (int, float)):
            return Matrix([[self._data[i][j] * float(other) for j in range(self.cols)] for i in range(self.rows)])
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Incompatible shapes for multiplication: {self.shape} × {other.shape}")
            result = [
                [sum(self._data[i][k] * other._data[k][j] for k in range(self.cols)) for j in range(other.cols)]
                for i in range(self.rows)
            ]
            return Matrix(result)
        return NotImplemented

    def __rmul__(self, other: Number) -> "Matrix":
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def det(self) -> float:
        if self.rows != self.cols:
            raise ValueError("Determinant is only defined for square matrices.")
        n = self.rows
        A = [row[:] for row in self._data]
        sign = 1.0
        for i in range(n):
            pivot = i
            max_val = abs(A[i][i])
            for r in range(i + 1, n):
                v = abs(A[r][i])
                if v > max_val:
                    max_val = v
                    pivot = r
            if max_val == 0.0:
                return 0.0
            if pivot != i:
                A[i], A[pivot] = A[pivot], A[i]
                sign *= -1.0
            pivot_val = A[i][i]
            for r in range(i + 1, n):
                factor = A[r][i] / pivot_val
                A[r][i] = 0.0
                for j in range(i + 1, n):
                    A[r][j] -= factor * A[i][j]
        det_val = sign
        for i in range(n):
            det_val *= A[i][i]
        return det_val


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())
            item.layout().deleteLater()


class MatrixPanelWidget(QGroupBox):
    def __init__(self, name: str, parent=None):
        super().__init__(f"Matrix {name}", parent)
        self.panel_name = name
        self.rows_data: List[List[float]] = []
        self.columns_locked = False

        self.setLayout(QVBoxLayout())
        # Row: columns selector
        cols_row = QHBoxLayout()
        cols_row.addWidget(QLabel("Columns (1–5):"))
        self.column_spin = QSpinBox()
        self.column_spin.setRange(1, 5)
        self.column_spin.setValue(3)
        self.column_spin.valueChanged.connect(lambda _: self.rebuild_row_inputs())
        cols_row.addWidget(self.column_spin)
        cols_row.addStretch(1)
        self.layout().addLayout(cols_row)

        # Row: next-row inputs
        self.row_input_frame = QFrame()
        self.row_input_layout = QHBoxLayout(self.row_input_frame)
        self.layout().addWidget(self.row_input_frame)
        self.row_entry_widgets: List[QLineEdit] = []
        self.rebuild_row_inputs()

        # Row: buttons + rows label
        btns_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Row")
        self.add_btn.clicked.connect(self.add_row)
        btns_row.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove Last Row")
        self.remove_btn.clicked.connect(self.remove_last_row)
        btns_row.addWidget(self.remove_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_rows)
        btns_row.addWidget(self.clear_btn)

        btns_row.addStretch(1)
        self.rows_label = QLabel("Rows: 0/5")
        btns_row.addWidget(self.rows_label)
        self.layout().addLayout(btns_row)

        # Matrix display grid
        self.matrix_display_frame = QFrame()
        self.matrix_grid = QGridLayout(self.matrix_display_frame)
        self.layout().addWidget(self.matrix_display_frame)

        self.refresh_display()

    def rebuild_row_inputs(self):
        clear_layout(self.row_input_layout)
        self.row_entry_widgets = []
        self.row_input_layout.addWidget(QLabel("Next row:"))
        column_count = self.column_spin.value()
        for _ in range(column_count):
            le = QLineEdit()
            le.setPlaceholderText("num")
            le.setFixedWidth(70)
            self.row_entry_widgets.append(le)
            self.row_input_layout.addWidget(le)
        self.row_input_layout.addStretch(1)

    def add_row(self):
        if len(self.rows_data) >= 5:
            QMessageBox.critical(self, "Limit reached", "Maximum 5 rows.")
            return
        column_count = self.column_spin.value()
        try:
            new_row = [float(le.text()) for le in self.row_entry_widgets]
        except ValueError:
            QMessageBox.critical(self, "Invalid input", "Please enter numeric values for the entire row.")
            return
        if not new_row or len(new_row) != column_count:
            QMessageBox.critical(self, "Invalid row", "Row must have exactly the selected number of columns.")
            return
        if self.rows_data and len(self.rows_data[0]) != column_count:
            QMessageBox.critical(self, "Size mismatch", "Number of columns cannot change after adding rows.")
            return
        self.rows_data.append(new_row)
        for le in self.row_entry_widgets:
            le.clear()
        self.refresh_display()
        if not self.columns_locked:
            self.column_spin.setDisabled(True)
            self.columns_locked = True

    def remove_last_row(self):
        if not self.rows_data:
            return
        self.rows_data.pop()
        self.refresh_display()
        if not self.rows_data and self.columns_locked:
            self.column_spin.setDisabled(False)
            self.columns_locked = False

    def clear_rows(self):
        self.rows_data.clear()
        self.refresh_display()
        if self.columns_locked:
            self.column_spin.setDisabled(False)
            self.columns_locked = False

    def refresh_display(self):
        clear_layout(self.matrix_grid)
        for r, row in enumerate(self.rows_data):
            for c, value in enumerate(row):
                lbl = QLabel(f"{value:.6g}")
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.matrix_grid.addWidget(lbl, r, c)
        self.rows_label.setText(f"Rows: {len(self.rows_data)}/5")

    def get_matrix(self) -> Matrix:
        if not self.rows_data:
            raise ValueError(f"Matrix {self.panel_name} is empty.")
        return Matrix(self.rows_data)

    def set_matrix(self, matrix_value: Matrix):
        self.rows_data = matrix_value.tolist()
        self.refresh_display()
        if self.rows_data:
            self.column_spin.setValue(len(self.rows_data[0]))
            self.column_spin.setDisabled(True)
            self.columns_locked = True
        else:
            self.column_spin.setDisabled(False)
            self.columns_locked = False


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Builder (PyQt, up to 5x5)")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Panels
        panels_row = QHBoxLayout()
        self.panel_A = MatrixPanelWidget("A")
        self.panel_B = MatrixPanelWidget("B")
        panels_row.addWidget(self.panel_A)
        panels_row.addWidget(self.panel_B)
        main_layout.addLayout(panels_row)

        # Controls (compact)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Operation:"))

        self.operations = [
            "A + B",
            "A - B",
            "B - A",
            "A × B",
            "B × A",
            "k × A",
            "k × B",
            "det(A)",
            "det(B)",
        ]
        self.op_combo = QComboBox()
        self.op_combo.addItems(self.operations)
        controls.addWidget(self.op_combo)

        controls.addSpacing(12)
        controls.addWidget(QLabel("k:"))
        self.scalar_edit = QLineEdit()
        self.scalar_edit.setPlaceholderText("scalar")
        self.scalar_edit.setFixedWidth(90)
        self.scalar_edit.returnPressed.connect(self.run_operation)
        controls.addWidget(self.scalar_edit)

        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.run_operation)
        controls.addWidget(run_btn)

        controls.addStretch(1)
        clear_btn = QPushButton("Clear Result")
        clear_btn.clicked.connect(lambda: self.display_result(""))
        controls.addWidget(clear_btn)

        main_layout.addLayout(controls)

        # Result
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.result_text.setMinimumHeight(220)
        main_layout.addWidget(self.result_text)
        self.display_result("")

    def display_result(self, content):
        self.result_text.setPlainText(str(content) if content is not None else "")

    def get_panel(self, name: str) -> MatrixPanelWidget:
        return self.panel_A if name == "A" else self.panel_B

    def get_matrix_or_error(self, name: str):
        try:
            return self.get_panel(name).get_matrix()
        except Exception as e:
            QMessageBox.critical(self, "Matrix error", str(e))
            return None

    def get_scalar_or_error(self):
        text = self.scalar_edit.text().strip()
        if not text:
            QMessageBox.critical(self, "Scalar error", "Please enter a scalar value k.")
            return None
        try:
            return float(text)
        except ValueError:
            QMessageBox.critical(self, "Scalar error", f"Invalid scalar: {text}")
            return None

    def run_operation(self):
        op = self.op_combo.currentText()
        try:
            if op in ("A + B", "A - B", "B - A", "A × B", "B × A"):
                A = self.get_matrix_or_error("A")
                B = self.get_matrix_or_error("B")
                if A is None or B is None:
                    return
                if op == "A + B":
                    self.display_result(A + B)
                elif op == "A - B":
                    self.display_result(A - B)
                elif op == "B - A":
                    self.display_result(B - A)
                elif op == "A × B":
                    self.display_result(A * B)
                elif op == "B × A":
                    self.display_result(B * A)
            elif op in ("k × A", "k × B"):
                k = self.get_scalar_or_error()
                if k is None:
                    return
                which = "A" if op.endswith("A") else "B"
                M = self.get_matrix_or_error(which)
                if M is None:
                    return
                self.display_result(k * M)
            elif op in ("det(A)", "det(B)"):
                which = "A" if op == "det(A)" else "B"
                M = self.get_matrix_or_error(which)
                if M is None:
                    return
                value = M.det()
                self.display_result(f"det({which}) = {value:.6g}")
        except Exception as e:
            QMessageBox.critical(self, "Operation error", str(e))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    w = AppWindow()
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec_())