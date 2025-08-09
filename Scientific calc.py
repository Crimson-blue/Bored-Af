import tkinter as tk
from tkinter import ttk
import math
import re

class ScientificCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Scientific Calculator")
        self.root.geometry("400x550")
        self.root.resizable(False, False)
        self.result_var = tk.StringVar(value="0")
        self.current_expression = ""
        self.create_gui()
        
    def create_gui(self):
        display_frame = tk.Frame(self.root)
        display_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="nsew")
        
        self.expr_label = tk.Label(display_frame, text="", anchor="e", bg="white", relief="sunken", font=("Arial", 10), height=1)
        self.expr_label.pack(fill="x", padx=2, pady=(2,0))
        display = ttk.Entry(display_frame, textvariable=self.result_var, justify="right", font=("Arial", 18), state="readonly")
        display.pack(fill="x", padx=2, pady=(0,2))
        buttons = [
            ("C", 1, 0), ("CE", 1, 1), ("⌫", 1, 2), ("/", 1, 3),
            ("sin", 2, 0), ("cos", 2, 1), ("tan", 2, 2), ("*", 2, 3),
            ("ln", 3, 0), ("log", 3, 1), ("√", 3, 2), ("-", 3, 3),
            ("(", 4, 0), (")", 4, 1), ("^", 4, 2), ("+", 4, 3),
            ("7", 5, 0), ("8", 5, 1), ("9", 5, 2), ("^2", 5, 3),
            ("4", 6, 0), ("5", 6, 1), ("6", 6, 2), ("^3", 6, 3),
            ("1", 7, 0), ("2", 7, 1), ("3", 7, 2), ("!", 7, 3),
            ("0", 8, 0), (".", 8, 1), ("π", 8, 2), ("e", 8, 3),
            ("=", 9, 0, 4)]
        
        for button in buttons:
            if len(button) == 3:  # Normal button
                text, row, col = button
                columnspan = 1
            else:  # Special case for equals button (4 values)
                text, row, col, columnspan = button
                
            btn = ttk.Button(self.root, text=text, 
                           command=lambda x=text: self.button_click(x))
            btn.grid(row=row, column=col, columnspan=columnspan, 
                    padx=2, pady=2, sticky="nsew")
            
        # Configure grid weights
        for i in range(10):
            self.root.grid_rowconfigure(i, weight=1)
        for i in range(4):
            self.root.grid_columnconfigure(i, weight=1)
    
    def safe_eval(self, expression):
        allowed_names = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log10, "ln": math.log, "sqrt": math.sqrt,
            "pi": math.pi, "e": math.e, "factorial": math.factorial,
            "abs": abs, "pow": pow, "__builtins__": {}
        }
        try:
            return eval(expression, allowed_names)
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")
    
    def preprocess_expression(self, expr):
        expr = expr.replace("π", "pi").replace("√", "sqrt")
        expr = expr.replace("^2", "**2").replace("^3", "**3")
        expr = expr.replace("^", "**")
        
        # Handle implicit multiplication (e.g., "2π" becomes "2*pi")
        expr = re.sub(r'(\d)(pi|e|sin|cos|tan|log|ln|sqrt)', r'\1*\2', expr)
        expr = re.sub(r'(pi|e)(\d)', r'\1*\2', expr)
        expr = re.sub(r'(\))(\d|pi|e|sin|cos|tan|log|ln|sqrt)', r'\1*\2', expr)
        expr = re.sub(r'(\d|pi|e)(\()', r'\1*\2', expr)
        
        return expr
    
    def button_click(self, value):
        current = self.result_var.get()
        
        if value == "C":
            self.result_var.set("0")
            self.current_expression = ""
            self.expr_label.config(text="")
        elif value == "CE":
            self.result_var.set("0")
        elif value == "⌫":
            if len(current) > 1:
                self.result_var.set(current[:-1])
                if self.current_expression:
                    self.current_expression = self.current_expression[:-1]
                    self.expr_label.config(text=self.current_expression)
            else:
                self.result_var.set("0")
                self.current_expression = ""
                self.expr_label.config(text="")
        elif value == "=":
            try:
                if self.current_expression:
                    expr = self.preprocess_expression(self.current_expression)
                    result = self.safe_eval(expr)
                    self.result_var.set(str(result))
                    self.current_expression = str(result)
                    self.expr_label.config(text="")
            except:
                self.result_var.set("Error")
                self.current_expression = ""
                self.expr_label.config(text="")
        elif value in ["^2", "^3"]:
            try:
                num = float(current)
                power = int(value[1])
                result = num ** power
                self.result_var.set(str(result))
                self.current_expression = str(result)
                self.expr_label.config(text="")
            except:
                self.result_var.set("Error")
        elif value == "!":
            try:
                num = float(current)
                if num >= 0 and num == int(num):
                    result = math.factorial(int(num))
                    self.result_var.set(str(result))
                    self.current_expression = str(result)
                    self.expr_label.config(text="")
                else:
                    self.result_var.set("Error")
            except:
                self.result_var.set("Error")
        elif value in ["sin", "cos", "tan", "log", "ln", "√"]:
            if current == "0" or current == "Error":
                self.current_expression = value + "("
            else:
                self.current_expression += value + "("
            self.result_var.set("0")
            self.expr_label.config(text=self.current_expression)
        elif value in ["+", "-", "*", "/", "^", "(", ")"]:
            if current == "Error":
                return
            if not self.current_expression and current != "0":
                self.current_expression = current
            self.current_expression += value
            self.expr_label.config(text=self.current_expression)
            if value != "(":
                self.result_var.set("0")
        else:
            if current == "0" or current == "Error":
                self.result_var.set(value)
                if not self.current_expression:
                    self.current_expression = value
                else:
                    self.current_expression += value
            else:
                self.result_var.set(current + value)
                self.current_expression += value
            self.expr_label.config(text=self.current_expression)

if __name__ == "__main__":
    root = tk.Tk()
    app = ScientificCalculator(root)
    root.mainloop()