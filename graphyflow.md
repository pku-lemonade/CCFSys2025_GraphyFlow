Project Path: CCFSys2025_GraphyFlow

Source Tree:

```txt
CCFSys2025_GraphyFlow
├── README.md
├── graphyflow
│   ├── __init__.py
│   ├── backend_defines.py
│   ├── backend_manager.py
│   ├── backend_utils.py
│   ├── dataflow_ir.py
│   ├── dataflow_ir_datatype.py
│   ├── global_graph.py
│   ├── graph_types.py
│   ├── hls_utils.py
│   ├── lambda_func.py
│   ├── passes.py
│   ├── project_generator.py
│   ├── project_template
│   │   ├── Makefile
│   │   ├── gen_random_graph.py
│   │   ├── global_para.mk
│   │   ├── run.sh
│   │   ├── scripts
│   │   │   ├── clean.mk
│   │   │   ├── help.mk
│   │   │   ├── host
│   │   │   │   ├── fpga_executor.cpp
│   │   │   │   ├── fpga_executor.h
│   │   │   │   ├── graph_loader.cpp
│   │   │   │   ├── graph_loader.h
│   │   │   │   ├── host.cpp
│   │   │   │   ├── host.mk
│   │   │   │   ├── host_bellman_ford.cpp
│   │   │   │   ├── host_bellman_ford.h
│   │   │   │   ├── host_verifier.cpp
│   │   │   │   ├── host_verifier.h
│   │   │   │   ├── xcl2.cpp
│   │   │   │   └── xcl2.h
│   │   │   ├── kernel
│   │   │   │   └── kernel.mk
│   │   │   ├── main.mk
│   │   │   └── utils.mk
│   │   └── system.cfg
│   ├── simulate.py
│   └── visualize_ir.py
└── tests
    └── bellman_ford.py

```

`CCFSys2025_GraphyFlow/README.md`:

```md
# GraphyFlow: A High-Level Synthesis Framework for Graph Computing on FPGAs

## 1\. Overview

GraphyFlow is a Python-based framework designed to simplify the development of graph algorithms for hardware acceleration on FPGAs. It provides a high-level, functional API for developers to express complex graph computations. The framework then automatically translates this high-level description into a Dataflow-Graph Intermediate Representation (DFG-IR), which is subsequently compiled into a complete, runnable Vitis HLS project, including HLS C++ for the kernel and C++ for the host application.

The primary goal of GraphyFlow is to abstract away the complexities of HLS and FPGA project management, allowing domain experts to focus on the algorithm's logic while still leveraging the performance of hardware acceleration.

## 2\. Core Concepts & Architecture

GraphyFlow employs a multi-stage compilation architecture to transform a high-level Python definition into a low-level hardware implementation.

```mermaid
graph TD
    A[<b>Step 1: Algorithm Definition</b><br/>User writes a graph algorithm in Python using the GraphyFlow API<br/><i>(e.g., tests/new_dist.py)</i>] --> B{<b>Step 2: Frontend Compilation</b><br/>The Python API builds a high-level graph representation<br/><i>(graphyflow/global_graph.py)</i>};
    B --> C[<b>Step 3: DFG-IR Generation</b><br/>The graph is converted into a<br/>Dataflow-Graph Intermediate Representation<br/><i>(graphyflow/dataflow_ir.py)</i>];
    C --> D{<b>Step 4: Backend Code Generation</b><br/>The Backend Manager traverses the DFG-IR to generate<br/>HLS C++, Host C++, and build scripts<br/><i>(graphyflow/backend_manager.py)</i>};
    D --> E[<b>Step 5: Project Assembly</b><br/>All generated and static template files<br/>are assembled into a complete Vitis project<br/><i>(graphyflow/project_generator.py)</i>];
```

## 3\. Prerequisites

To use GraphyFlow and build the generated projects, you will need the following software installed and configured on your system:

  * **Python 3.x**: For running the GraphyFlow framework itself.
  * **Xilinx Vitis**: The core toolchain for HLS and building the FPGA binaries. The project files seem to be configured for **Vitis 2022.2**, so using this version is recommended for compatibility.
  * **Xilinx Runtime (XRT)**: Required for communication between the host and the FPGA. This is typically installed with Vitis.
  * **Environment Variables**: Ensure that the Vitis and XRT environment setup scripts have been sourced (e.g., `source /opt/Xilinx/Vitis/2022.2/settings64.sh` and `source /opt/xilinx/xrt/setup.sh`).

## 4\. Directory Structure

The GraphyFlow repository is organized as follows:

```
GraphyFlow/
├── graphyflow/              # Core framework source code
│   ├── backend_manager.py     # The main backend compiler
│   ├── dataflow_ir.py         # Defines the Intermediate Representation
│   ├── global_graph.py        # The user-facing Python API for algorithm definition
│   ├── project_generator.py   # Assembles the final Vitis project
│   └── project_template/      # Static template files for a Vitis project
├── tests/                   # Example scripts demonstrating how to use GraphyFlow
│   ├── new_dist.py            # The primary example for generating a project
│   └── ...
└── README.md                # This README file
```

## 5\. How to Run the Example

The following steps guide you through generating, building, and running the provided distance computation project.

### Step 1: Generate the Vitis Project

The framework uses the pre-defined algorithm in `tests/new_dist.py` to generate a complete Vitis project. This script begins by defining the data structure of the graph's nodes and edges for the compiler:

```python
# tests/new_dist.py (Snippet)
# --- Define graph properties ---
g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)
```

To generate the project, execute the following command from the root of the `GraphyFlow` directory:

```bash
PYTHONPATH=$(pwd) python3 tests/new_dist.py
```

This command temporarily adds the `GraphyFlow` project root to your Python path, allowing the script to import the necessary framework modules. After it completes, a new directory `generated_project/` will be created.

### Step 2: Build the FPGA Kernel and Host Executable

Navigate into the generated project directory and use the provided `Makefile` to build the project. You must specify a target platform.

```bash
cd generated_project/
make check TARGET=sw_emu
```

  * The `make check` command is a convenient wrapper that first builds everything (`all`) and then runs the simulation (`run.sh`).
  * `TARGET` can be one of the following:
      * `sw_emu`: Software emulation (fastest build).
      * `hw_emu`: Hardware emulation (more accurate, slower build).
      * `hw`: Full hardware synthesis (very slow, for running on the actual FPGA).

### Step 3: Run the Project

If you built the project using `make all` instead of `make check`, you can run the software emulation manually:

```bash
./run.sh sw_emu
```

The script will execute the host program, which loads the generated FPGA binary (`.xclbin`), prepares a sample graph, runs the computation, verifies the result against a CPU implementation, and prints a success or failure message.

## 6\. Algorithm Definition API

You can define graph algorithms by chaining together a series of high-level dataflow operators.

  * `n.map_(map_func)`

      * **Description**: Applies a lambda function `map_func` to each element of the input data array `n`.
      * **Lambda Input**: A single element from the array `n`.
      * **Lambda Output**: The transformed element.

  * `n.filter(filter_func)`

      * **Description**: Filters the input array `n`, keeping only the elements for which `filter_func` returns `True`.
      * **Lambda Input**: A single element from the array `n`.
      * **Lambda Output**: A boolean value.

  * `n.reduce_by(reduce_key, reduce_transform, reduce_method)`

      * **Description**: A powerful operator that performs a grouped reduction.
      * **`reduce_key`**: A lambda that computes a key for each element. Elements with the same key are grouped together.
      * **`reduce_transform`**: A lambda that transforms each element within a group before reduction.
      * **`reduce_method`**: A lambda that takes two transformed elements and combines them into one. This function must be commutative and associative (e.g., `add`, `min`, `max`). The final output is an array containing one reduced value for each group.

```

`CCFSys2025_GraphyFlow/graphyflow/backend_defines.py`:

```py
# This is a new backend 'cause the old backend's code is too messy
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


INDENT_UNIT = "    "


class HLSBasicType(Enum):
    UINT = "uint32_t"
    UINT8 = "uint8_t"
    UINT16 = "uint16_t"
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    AP_FIXED_POD = "int32_t"
    REAL_FLOAT = "float"
    BOOL = "bool"
    STRUCT = "struct"
    STREAM = "stream"
    ARRAY = "array"
    POINTER = "pointer"

    def __repr__(self) -> str:
        return self.value

    @property
    def is_simple(self) -> bool:
        return self not in [
            HLSBasicType.STRUCT,
            HLSBasicType.STREAM,
            HLSBasicType.ARRAY,
            HLSBasicType.POINTER,
        ]


class HLSType:
    _all_full_names = set()
    _all_names = set()
    _full_to_type = {}
    _name_to_full = {}
    _id_cnt = 0

    def __init__(
        self,
        basic_type: HLSBasicType,
        sub_types: Optional[List[HLSType]] = None,
        struct_name: Optional[str] = None,
        struct_prop_names: Optional[List[str]] = None,
        array_dims: Optional[List[Union[str, int]]] = None,
        is_const_ptr: bool = False,
    ) -> None:
        self.type = basic_type
        self.sub_types = sub_types
        self.readable_id = HLSType._id_cnt
        self.struct_prop_names = None
        self.array_dims = array_dims
        self.is_const_ptr = is_const_ptr

        if basic_type.is_simple:
            self.name = basic_type.value
            self.full_name = self.name
        elif basic_type == HLSBasicType.STREAM:
            assert sub_types and len(sub_types) == 1
            self.name = f"hls::stream<{sub_types[0].name}>"
            self.full_name = f"hls::stream<{sub_types[0].full_name}>"
        elif basic_type == HLSBasicType.ARRAY:
            assert sub_types and len(sub_types) == 1 and array_dims and len(array_dims) > 0
            dims_str = "".join(f"[{d}]" for d in self.array_dims)
            self.name = f"{sub_types[0].name}{dims_str}"
            self.full_name = f"{sub_types[0].full_name}{dims_str}"
        elif basic_type == HLSBasicType.POINTER:
            assert sub_types and len(sub_types) == 1
            const_str = "const " if self.is_const_ptr else ""
            self.name = f"{const_str}{sub_types[0].name}*"
            self.full_name = f"{const_str}{sub_types[0].full_name}*"
        elif basic_type == HLSBasicType.STRUCT:
            assert sub_types and len(sub_types) > 0
            self.full_name = self._generate_canonical_name(sub_types, explicit_name=struct_name)

            if self.full_name in HLSType._all_full_names:
                existing_type = HLSType._full_to_type[self.full_name]
                self.__dict__.update(existing_type.__dict__)
                return

            self.name = struct_name if struct_name else self._generate_readable_name(sub_types)

            if struct_prop_names:
                assert len(struct_prop_names) == len(sub_types)
                self.struct_prop_names = struct_prop_names
        else:
            assert False, f"Basic type {basic_type} not supported"

        # Caching for truly new types
        if self.full_name in HLSType._all_full_names:
            existing_type = HLSType._full_to_type[self.full_name]
            self.__dict__.update(existing_type.__dict__)
            return

        HLSType._all_full_names.add(self.full_name)

        # --- *** 关键修正：仅对非简单类型进行名称冲突检查 *** ---
        if not self.type.is_simple:
            if self.name in HLSType._all_names:
                if struct_name is not None:
                    assert False, f"Struct name collision detected: {self.name}"
                else:
                    self.name = f"{self.name}_{self.readable_id}"

        HLSType._all_names.add(self.name)
        HLSType._full_to_type[self.full_name] = self
        HLSType._name_to_full[self.name] = self.full_name
        HLSType._id_cnt += 1

    @classmethod
    def get_type(cls, type_name):
        assert type_name in cls._all_names
        return cls._full_to_type[cls._name_to_full[type_name]]

    def _generate_canonical_name(self, sub_types: List[HLSType], explicit_name: Optional[str] = None) -> str:
        name_parts = [t.full_name.replace(" ", "_").replace("*", "_ptr") for t in sub_types]
        if explicit_name:
            name_parts.insert(0, explicit_name)
        return f"struct_{'_'.join(name_parts)}_t"

    def _generate_readable_name(self, sub_types: List[HLSType]) -> str:
        name_parts = [t.name[:1] for t in sub_types]
        return f"struct_{''.join(name_parts)}_{self.readable_id}_t"

    def get_nth_subname(self, n: int):
        assert self.sub_types
        if not self.struct_prop_names:
            member_names = [f"ele_{i}" for i in range(len(self.sub_types))]
        else:
            member_names = self.struct_prop_names
        assert n < len(member_names)
        return member_names[n]

    def get_upper_decl(self, var_name: str):
        """Get decl for upper struct"""
        if self.type == HLSBasicType.ARRAY:
            match = re.search(r"\[", self.name)
            if match:
                base_type_str = self.name[: match.start()]
                dims_str = self.name[match.start() :]
                return f"{base_type_str} {var_name}{dims_str}"
            else:
                assert False
                # return f"{self.name} {var_name};"
        return f"{self.name} {var_name}"

    def get_upper_param(self, var_name: str, ref: bool):
        if self.type == HLSBasicType.ARRAY:
            match = re.search(r"\[", self.name)
            if match:
                base_type_str = self.name[: match.start()]
                dims_str = self.name[match.start() :]
                if ref:
                    return f"{base_type_str} (&{var_name}){dims_str}"
                return f"{base_type_str} {var_name}{dims_str}"
            else:
                assert False
        if ref:
            return f"{self.name} &{var_name}"
        return f"{self.name} {var_name}"

    def gen_decl(self, member_names: Optional[List[str]] = None) -> str:
        # Generate C++ typedef struct declaration
        assert self.type == HLSBasicType.STRUCT
        if member_names is None:
            if self.struct_prop_names:
                member_names = self.struct_prop_names
            else:
                member_names = [f"ele_{i}" for i in range(len(self.sub_types))]

        assert len(member_names) == len(self.sub_types)
        if self.struct_prop_names:
            assert self.struct_prop_names == member_names
        decls = [st.get_upper_decl(m_name) + ";" for st, m_name in zip(self.sub_types, member_names)]

        return (
            f"struct __attribute__((packed)) {self.name} {{\n"
            + f"\n".join([INDENT_UNIT + d for d in decls])
            + f"\n}};\n"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLSType):
            return NotImplemented
        return self.name == other.name and self.full_name == other.full_name

    def __hash__(self) -> int:
        return hash(self.full_name)

    def __repr__(self) -> str:
        return self.name


class HLSVar:
    def __init__(self, var_name: str, var_type: HLSType) -> None:
        self.name = var_name
        self.type = var_type

    def __repr__(self) -> str:
        return f"HLSVar({self.name}, {self.type})"


class HLSCodeLine:
    def __init__(self) -> None:
        pass

    def gen_code(self, indent_lvl: int = 0) -> str:
        assert False, "This function shouldn't be called"


class CodeVarDecl(HLSCodeLine):
    def __init__(self, var_name, var_type, init_val=None) -> None:
        super().__init__()
        self.var = HLSVar(var_name, var_type)
        self.init_val = init_val

    def gen_code(self, indent_lvl: int = 0):
        init_code = f" = {self.init_val}" if self.init_val is not None else ""
        return indent_lvl * INDENT_UNIT + self.var.type.get_upper_decl(self.var.name) + init_code + ";\n"


class CodeIf(HLSCodeLine):
    def __init__(
        self,
        expr: HLSExpr,
        if_codes: List[HLSCodeLine],
        else_codes: List[HLSCodeLine] = None,
        elifs: List[Tuple[HLSExpr, List[HLSCodeLine]]] = None,
    ) -> None:
        super().__init__()
        if type(expr) == HLSVar:
            expr = HLSExpr(HLSExprT.VAR, expr)
        self.expr = expr
        self.if_codes = if_codes
        self.elifs = elifs if elifs else []
        self.else_codes = else_codes if else_codes else []

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        if_part = (
            oind
            + "if ("
            + self.expr.code
            + ") {\n"
            + "".join(c.gen_code(indent_lvl + 1) for c in self.if_codes)
            + oind
            + "}"
        )
        elif_part = ""
        for expr, codes in self.elifs:
            elif_part += (
                f" else if ({expr.code}) "
                + "{\n"
                + "".join(c.gen_code(indent_lvl + 1) for c in codes)
                + oind
                + "}"
            )
        else_part = ""
        if self.else_codes:
            else_part = (
                " else {\n" + "".join(c.gen_code(indent_lvl + 1) for c in self.else_codes) + oind + "}"
            )
        return if_part + elif_part + else_part + "\n"


class CodeWhile(HLSCodeLine):
    def __init__(
        self,
        codes: List[HLSCodeLine],
        iter_expr: HLSExpr,
    ) -> None:
        super().__init__()
        self.i_expr = iter_expr
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return (
            oind
            + f"while ({self.i_expr.code}) "
            + "{\n"
            + "".join(c.gen_code(indent_lvl + 1) for c in self.codes)
            + oind
            + "}\n"
        )


class CodeFor(HLSCodeLine):
    def __init__(
        self,
        codes: List[HLSCodeLine],
        iter_limit: Union[str, HLSVar],
        iter_cmp="<",
        iter_name="i",
    ) -> None:
        super().__init__()
        self.i_name = iter_name
        self.i_cmp = iter_cmp
        self.i_lim = iter_limit
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return (
            oind
            + f"for (uint32_t {self.i_name} = 0; {self.i_name} {self.i_cmp} {self.i_lim}; {self.i_name}++) "
            + "{\n"
            + "".join(c.gen_code(indent_lvl + 1) for c in self.codes)
            + oind
            + "}\n"
        )


class CodeBreak(HLSCodeLine):
    def __init__(self) -> None:
        super().__init__()

    def gen_code(self, indent_lvl: int = 0) -> str:
        return indent_lvl * INDENT_UNIT + "break;\n"


class HLSExprT(Enum):
    CONST = "const"
    VAR = "var"
    UOP = "uop"
    BINOP = "binop"
    STREAM_READ = "stream_read"
    STREAM_EMPTY = "stream_empty"


class HLSExpr:
    def __init__(
        self,
        expr_type: HLSExprT,
        expr_val: Any,
        operands: Optional[List[HLSExpr]] = None,
    ) -> None:
        if expr_type == HLSExprT.CONST:
            assert type(expr_val) in [int, float, bool, str]
            assert operands is None
        elif expr_type == HLSExprT.VAR:
            assert type(expr_val) == HLSVar
            assert operands is None
        elif expr_type == HLSExprT.STREAM_READ:
            assert expr_val is None
            assert type(operands) == list and len(operands) == 1
            assert operands[0].type == HLSExprT.VAR
        elif expr_type == HLSExprT.STREAM_EMPTY:
            assert expr_val is None
            assert type(operands) == list and len(operands) == 1
            assert operands[0].type == HLSExprT.VAR
        elif expr_type == HLSExprT.UOP:
            if type(expr_val) == tuple:
                assert type(expr_val[0]) == dfir.UnaryOp
            else:
                assert type(expr_val) == dfir.UnaryOp
            assert type(operands) == list and len(operands) == 1
        elif expr_type == HLSExprT.BINOP:
            assert type(expr_val) == dfir.BinOp
            assert type(operands) == list and len(operands) == 2
        else:
            assert False, f"Type {expr_type} and val {expr_val} not supported"
        self.type = expr_type
        self.val = expr_val
        self.operands = operands

    @classmethod
    def check_const(cls, hls_expr: HLSExpr, port: dfir.Port):
        if port.from_const:
            return HLSExpr(HLSExprT.CONST, port.from_const_val)
        return hls_expr

    @property
    def contain_s_read(self) -> bool:
        if self.type in [HLSExprT.CONST, HLSExprT.VAR]:
            return False
        elif self.type in [HLSExprT.STREAM_READ, HLSExprT.STREAM_EMPTY]:
            return True
        elif self.type in [HLSExprT.UOP, HLSExprT.BINOP]:
            return any(opr.contain_s_read for opr in self.operands)
        else:
            assert False, f"Type {self.type} not supported"

    @property
    def code(self) -> str:
        if self.type == HLSExprT.CONST:
            if type(self.val) == float:
                return f"(({HLSBasicType.FLOAT.value}){self.val})"
            elif type(self.val) == bool:
                return "true" if self.val else "false"
            return str(self.val)
        elif self.type == HLSExprT.VAR:
            return self.val.name
        elif self.type == HLSExprT.STREAM_READ:
            return f"{self.operands[0].val.name}.read()"
        elif self.type == HLSExprT.STREAM_EMPTY:
            return f"{self.operands[0].val.name}.empty()"
        elif self.type == HLSExprT.UOP:
            trans_dict = {
                dfir.UnaryOp.NOT: "(!operand)",
                dfir.UnaryOp.NEG: "(-operand)",
                dfir.UnaryOp.CAST_BOOL: f"(({HLSBasicType.BOOL.value})(operand))",
                dfir.UnaryOp.CAST_INT: f"(({HLSBasicType.INT.value})(operand))",
                dfir.UnaryOp.CAST_FLOAT: f"(({HLSBasicType.FLOAT.value})(operand))",
                dfir.UnaryOp.SELECT: f"operand.ele_{self.val[1] if type(self.val) == tuple else '0'}",
                dfir.UnaryOp.GET_ATTR: f"operand.{self.val[1] if type(self.val) == tuple else 'ele_0'}",
            }
            if type(self.val) == tuple:
                expr_val_val = self.val[0]
            else:
                expr_val_val = self.val
            return trans_dict[expr_val_val].replace("operand", self.operands[0].code)
        elif self.type == HLSExprT.BINOP:
            if self.val in [dfir.BinOp.MAX, dfir.BinOp.MIN]:
                assert not self.contain_s_read
            return self.val.gen_repr_forbkd(self.operands[0].code, self.operands[1].code)
        else:
            assert False, f"Type {self.type} not supported"


class CodeAssign(HLSCodeLine):
    def __init__(self, var: HLSVar, expr: HLSExpr) -> None:
        super().__init__()
        assert type(var) == HLSVar
        if type(expr) == HLSVar:
            expr = HLSExpr(HLSExprT.VAR, expr)
        self.var = var
        self.expr = expr

    def gen_code(self, indent_lvl: int = 0) -> str:
        return INDENT_UNIT * indent_lvl + f"{self.var.name} = {self.expr.code};\n"


class CodeCall(HLSCodeLine):
    def __init__(self, func: HLSFunction, params: List[HLSVar]) -> None:
        super().__init__()
        self.func = func
        assert type(params) == list
        self.params = params
        assert len(func.params) == len(params), f"{func.params} != {params}"

    def gen_code(self, indent_lvl: int = 0) -> str:
        def get_name(param):
            if isinstance(param, HLSVar):
                return param.name
            elif isinstance(param, HLSExpr):
                return param.code
            else:
                assert False

        return (
            INDENT_UNIT * indent_lvl
            + f"{self.func.name}("
            + ", ".join(get_name(var) for var in self.params)
            + ");\n"
        )


class CodeWriteStream(HLSCodeLine):
    def __init__(self, stream_var: HLSVar, in_expr: Union[HLSVar, HLSExpr]) -> None:
        super().__init__()
        self.stream_var = stream_var
        if type(in_expr) == HLSVar:
            in_expr = HLSExpr(HLSExprT.VAR, in_expr)
        self.in_expr = in_expr

    def gen_code(self, indent_lvl: int = 0) -> str:
        return INDENT_UNIT * indent_lvl + f"{self.stream_var.name}.write({self.in_expr.code});\n"


class CodePragma(HLSCodeLine):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content

    def gen_code(self, indent_lvl: int = 0) -> str:
        return f"#pragma HLS {self.content}\n"


class CodeBlock(HLSCodeLine):
    """Represents a simple code block enclosed in braces."""

    def __init__(self, codes: List[HLSCodeLine]) -> None:
        super().__init__()
        self.codes = codes

    def gen_code(self, indent_lvl: int = 0) -> str:
        oind = indent_lvl * INDENT_UNIT
        return oind + "{\n" + "".join(c.gen_code(indent_lvl + 1) for c in self.codes) + oind + "}\n"


class CodeComment(HLSCodeLine):
    def __init__(self, text: str) -> None:
        super().__init__()
        assert "\n" not in text
        self.text = text

    def gen_code(self, indent_lvl: int = 0) -> str:
        return indent_lvl * INDENT_UNIT + "// " + self.text.strip() + "\n"


class HLSFunction:
    _readable_id_cnt = 0

    def __init__(
        self,
        name: str,
        comp: dfir.Component,
    ) -> None:
        self.name = name
        self.readable_id = HLSFunction._readable_id_cnt
        HLSFunction._readable_id_cnt += 1
        self.dfir_comp = comp
        self.params: List[HLSVar] = []
        self.codes: List[HLSCodeLine] = []
        # By default, a function is a standard streaming dataflow block.
        # This will be set to False for reduce sub-functions.
        self.streamed = True

    def __repr__(self) -> str:
        return f"HLSFunction({self.name}, {self.dfir_comp.name}, {self.dfir_comp.in_ports}, {self.params})"

```

`CCFSys2025_GraphyFlow/graphyflow/backend_manager.py`:

```py
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


from graphyflow.dataflow_ir import BinOp, UnaryOp
from graphyflow.backend_defines import (
    INDENT_UNIT,
    HLSType,
    HLSBasicType,
    HLSFunction,
    HLSCodeLine,
    HLSExpr,
    HLSExprT,
    HLSVar,
    CodeAssign,
    CodeBlock,
    CodeBreak,
    CodeCall,
    CodeComment,
    CodeFor,
    CodeIf,
    CodePragma,
    CodeVarDecl,
    CodeWhile,
    CodeWriteStream,
)
from graphyflow.backend_utils import generate_demux, generate_omega_network, generate_stream_zipper


class BackendManager:
    """Manages the entire HLS code generation process from a ComponentCollection."""

    def __init__(self):
        self.PE_NUM = 8
        self.STREAM_DEPTH = 8
        self.MAX_NUM = 16384  # For ReduceComponent key_mem size
        self.L = 4  # For ReduceComponent buffer size
        # Mappings to store results of type analysis
        self.type_map: Dict[dftype.DfirType, HLSType] = {}
        self.batch_type_map: Dict[HLSType, HLSType] = {}
        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}
        self.unstreamed_funcs: set[str] = set()

        # State for Phase 2 & 3
        self.hls_functions: Dict[int, HLSFunction] = {}
        self.top_level_stream_decls: List[Tuple[CodeVarDecl, CodePragma]] = []
        self.top_level_io_ports: List[dfir.Port] = []

        self.reduce_internal_streams: Dict[int, Dict[str, HLSVar]] = {}

        self.utility_functions: List[HLSFunction] = []
        self.reduce_helpers: Dict[int, Dict[str, Any]] = {}

        self.global_graph_store = None
        self.comp_col_store = None

        # 新增: 存储AXI相关的函数和类型信息
        self.axi_input_ports: List[dfir.Port] = []
        self.axi_output_ports: List[dfir.Port] = []
        self.mem_to_stream_func: Optional[HLSFunction] = None
        self.stream_to_mem_func: Optional[HLSFunction] = None
        self.dataflow_core_func: Optional[HLSFunction] = None

    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        """
        Main entry point to generate HLS header and source files.
        """
        self.global_graph_store = global_graph
        self.comp_col_store = comp_col
        header_name = f"{top_func_name}.h"

        # 1. Discover top-level I/O ports
        top_level_inputs = []
        for comp in comp_col.components:
            if isinstance(comp, dfir.IOComponent) and comp.io_type == dfir.IOComponent.IOType.INPUT:
                if comp.get_port("o_0").connected:
                    top_level_inputs.append(comp.get_port("o_0").connection)

        self.axi_input_ports = top_level_inputs
        self.axi_output_ports = comp_col.outputs
        self.top_level_io_ports = self.axi_input_ports + self.axi_output_ports

        # Phase 1: Type Analysis
        self._analyze_and_map_types(comp_col)
        # Phase 2: Function Definition and Stream Instantiation
        self._define_functions_and_streams(comp_col, top_func_name)
        # Phase 3: Code Body Generation
        self._translate_functions()

        # Phase 4: AXI Wrapper Generation
        self.mem_to_stream_func = self._generate_mem_to_stream_func()
        self.stream_to_mem_func = self._generate_stream_to_mem_func()
        self.dataflow_core_func = self._generate_dataflow_core_func(top_func_name)

        axi_wrapper_func_str, top_func_sig = self._generate_axi_kernel_wrapper(top_func_name)

        # 2. Generate file contents
        header_code = self._generate_header_file(top_func_name, top_func_sig)
        source_code = self._generate_source_file(header_name, axi_wrapper_func_str)

        return header_code, source_code

    def generate_host_codes(self, top_func_name: str, template_path: Path) -> Tuple[str, str]:
        """
        Generates host C++ files by filling in a new, more detailed template.
        This version is adapted for iterative algorithms like Bellman-Ford and
        implements correct graph-to-batch packing logic.
        """
        # This function requires significant changes to inject the helper
        # and modify the buffer packing logic.
        h_template_path = template_path / "generated_host.h.template"
        cpp_template_path = template_path / "generated_host.cpp.template"

        with open(h_template_path, "r") as f:
            h_template = f.read()
        with open(cpp_template_path, "r") as f:
            cpp_template = f.read()

        # --- Initialize Snippets for Declarations ---
        host_buffer_decls = []
        device_buffer_decls = []

        # --- AXI Input/Output Port Analysis ---
        assert len(self.axi_input_ports) == 1, "Host generator expects one AXI input port."
        assert len(self.axi_output_ports) == 1, "Host generator expects one AXI output port."

        input_port = self.axi_input_ports[0]
        output_port = self.axi_output_ports[0]
        input_var_name = input_port.unique_name
        output_var_name = output_port.unique_name
        input_batch_type = self.batch_type_map[self.type_map[input_port.data_type]]
        output_host_type = self._get_host_output_type()

        # --- Generate Header Declarations ---
        host_buffer_decls.append(
            f"// These now use the POD versions of the structs defined in common.h\n    "
            f"std::vector<{input_batch_type.name}, aligned_allocator<{input_batch_type.name}>> h_{input_var_name};"
        )
        device_buffer_decls.append(f"cl::Buffer d_{input_var_name};")
        host_buffer_decls.append(
            f"std::vector<{output_host_type.name}, aligned_allocator<{output_host_type.name}>> h_{output_var_name};"
        )
        device_buffer_decls.append(f"cl::Buffer d_{output_var_name};")
        host_buffer_decls.append("std::vector<int, aligned_allocator<int>> h_stop_flag;")
        device_buffer_decls.append("cl::Buffer d_stop_flag;")

        # --- Generate CPP Implementation Strings ---
        helper_func = """
// Helper function to convert ap_fixed to int32_t by reinterpreting its bits
static int32_t ap_fixed_to_int32(const ap_fixed<32, 16>& val) {
    return *reinterpret_cast<const int32_t*>(&val);
}
"""
        setup_buffers_impl = f"""
void AlgorithmHost::setup_buffers(const GraphCSR &graph, int start_node) {{
    m_num_vertices = graph.num_vertices;
    cl_int err;
    
    h_distances.assign(m_num_vertices, INFINITY_DIST);
    if (start_node < m_num_vertices) {{ h_distances[start_node] = 0; }}

    h_{input_var_name}.clear();
    {input_batch_type.name} current_batch;
    int edges_in_batch = 0;

    for (int u = 0; u < graph.num_vertices; ++u) {{
        for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {{
            int v = graph.columns[i];
            int w = graph.weights[i];

            edge_t edge; // This is now the POD version of edge_t
            edge.src.id = u;
            edge.src.distance = ap_fixed_to_int32(h_distances[u]);
            edge.dst.id = v;
            edge.dst.distance = ap_fixed_to_int32(h_distances[v]);
            edge.weight = ap_fixed_to_int32(ap_fixed<32, 16>(w));

            current_batch.data[edges_in_batch] = edge;
            edges_in_batch++;

            if (edges_in_batch == PE_NUM) {{
                current_batch.end_pos = PE_NUM;
                current_batch.end_flag = false;
                h_{input_var_name}.push_back(current_batch);
                edges_in_batch = 0;
            }}
        }}
    }}

    if (edges_in_batch > 0) {{
        current_batch.end_pos = edges_in_batch;
        current_batch.end_flag = false;
        h_{input_var_name}.push_back(current_batch);
    }}
    
    if (!h_{input_var_name}.empty()) {{
        h_{input_var_name}.back().end_flag = true;
    }}

    m_num_batches = h_{input_var_name}.size();
    h_{output_var_name}.resize(m_num_batches);
    h_stop_flag.resize(1);

    OCL_CHECK(err, d_{input_var_name} = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, h_{input_var_name}.size() * sizeof({input_batch_type.name}), h_{input_var_name}.data(), &err));
    OCL_CHECK(err, d_{output_var_name} = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, h_{output_var_name}.size() * sizeof({output_host_type.name}), h_{output_var_name}.data(), &err));
    OCL_CHECK(err, d_stop_flag = cl::Buffer(m_context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeof(int), h_stop_flag.data(), &err));
}}
"""
        transfer_to_fpga_impl = f"""
void AlgorithmHost::transfer_data_to_fpga() {{
    cl_int err;
    h_stop_flag[0] = 0; // Reset stop flag before each iteration
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects({{d_{input_var_name}, d_stop_flag}}, 0));
}}
"""
        execute_kernel_impl = f"""
void AlgorithmHost::execute_kernel_iteration(cl::Event &event) {{
    cl_int err;
    int arg_idx = 0;
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_{input_var_name}));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_{output_var_name}));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_stop_flag));
    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, (uint16_t)m_num_batches));
    OCL_CHECK(err, err = m_q.enqueueTask(m_kernel, nullptr, &event));
}}
"""
        transfer_from_fpga_impl = f"""
void AlgorithmHost::transfer_data_from_fpga() {{
    cl_int err;
    OCL_CHECK(err, err = m_q.enqueueMigrateMemObjects({{d_{output_var_name}, d_stop_flag}}, CL_MIGRATE_MEM_OBJECT_HOST));
    m_q.finish(); // Ensure data is synced back to host
}}
"""
        convergence_impl = f"""
bool AlgorithmHost::check_convergence_and_update() {{
    bool changed = false;
    std::map<int, ap_fixed<32, 16>> min_distances;

    for (const auto& batch : h_{output_var_name}) {{
        for (int i = 0; i < batch.end_pos; ++i) {{
            int node_id = batch.data[i].id;
            ap_fixed<32, 16> dist = batch.data[i].distance;
            if (min_distances.find(node_id) == min_distances.end() || dist < min_distances[node_id]) {{
                min_distances[node_id] = dist;
            }}
        }}
    }}

    for (auto const &[node_id, new_dist] : min_distances) {{
        if (new_dist < h_distances[node_id]) {{
            h_distances[node_id] = new_dist;
            changed = true;
        }}
    }}

    if (changed) {{
        for (auto& batch : h_{input_var_name}) {{
            for (int i = 0; i < batch.end_pos; ++i) {{
                batch.data[i].src.distance = ap_fixed_to_int32(h_distances[batch.data[i].src.id]);
                batch.data[i].dst.distance = ap_fixed_to_int32(h_distances[batch.data[i].dst.id]);
            }}
        }}
    }}

    return !changed;
}}

const std::vector<int> &AlgorithmHost::get_results() const {{
    static std::vector<int> final_distances;
    final_distances.clear();
    final_distances.reserve(h_distances.size());
    for (const auto &dist : h_distances) {{
        if (dist > std::numeric_limits<int>::max()) {{
            final_distances.push_back(std::numeric_limits<int>::max());
        }} else {{
            final_distances.push_back(dist.to_int());
        }}
    }}
    return final_distances;
}}
"""

        # --- Replace Placeholders in Templates ---
        cpp_final = cpp_template
        cpp_final = cpp_final.replace("// {{GRAPHYFLOW_HELPER_FUNCTIONS}}", helper_func)
        cpp_final = cpp_final.replace("// {{GRAPHYFLOW_SETUP_BUFFERS_IMPL}}", setup_buffers_impl)
        cpp_final = cpp_final.replace("// {{GRAPHYFLOW_TRANSFER_TO_FPGA_IMPL}}", transfer_to_fpga_impl)
        cpp_final = cpp_final.replace("// {{GRAPHYFLOW_EXECUTE_KERNEL_IMPL}}", execute_kernel_impl)
        cpp_final = cpp_final.replace("// {{GRAPHYFLOW_TRANSFER_FROM_FPGA_IMPL}}", transfer_from_fpga_impl)

        start_str = "bool AlgorithmHost::check_convergence_and_update() {"
        end_str = "return final_distances;\n}"
        cpp_final_start = cpp_final.find(start_str)
        cpp_final_end = cpp_final.find(end_str) + len(end_str)

        if cpp_final_start != -1 and cpp_final_end != -1:
            cpp_final = cpp_final[:cpp_final_start] + convergence_impl + cpp_final[cpp_final_end:]

        h_final = h_template
        h_final = h_final.replace(
            "// {{GRAPHYFLOW_HOST_BUFFER_DECLARATIONS}}", "\n    ".join(host_buffer_decls)
        )
        h_final = h_final.replace(
            "// {{GRAPHYFLOW_DEVICE_BUFFER_DECLARATIONS}}", "\n    ".join(device_buffer_decls)
        )
        h_final = h_final.replace(
            "// {{GRAPHYFLOW_HOST_STATE_DECLARATIONS}}",
            "// This can remain as ap_fixed since it's only used for host-side logic.\n    "
            "std::vector<ap_fixed<32, 16>> h_distances;",
        )

        return h_final, cpp_final.strip()

    def generate_build_system_files(self, kernel_name: str, executable_name: str) -> Dict[str, str]:
        """
        Generates all necessary build and execution files.
        This version is updated based on the more complete `tmp_work` example.
        """
        files = {}

        # Makefile
        files[
            "Makefile"
        ] = f"""
# This file is auto-generated by GraphyFlow.
KERNEL_NAME := {kernel_name}
EXECUTABLE := {executable_name}

TARGET ?= sw_emu
DEVICE := /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm

.PHONY: all clean cleanall exe check

include scripts/main.mk

check: all
	./run.sh $(TARGET)
"""

        # run.sh
        files[
            "run.sh"
        ] = f"""
#!/bin/bash
# This file is auto-generated by GraphyFlow.
TARGET=${{1:-sw_emu}} # Default to sw_emu if no argument is given
EXECUTABLE="{executable_name}"
KERNEL="{kernel_name}"

echo "--- Running for target: $TARGET ---"

# 1. Setup Environment
source /opt/xilinx/xrt/setup.sh
source /opt/Xilinx/Vitis/2022.2/settings64.sh # Adjust to your Vitis version

if [ "$TARGET" = "sw_emu" ] || [ "$TARGET" = "hw_emu" ]; then
    export XCL_EMULATION_MODE=$TARGET
else
    unset XCL_EMULATION_MODE
fi

# 2. Find xclbin
XCLBIN_FILE="./xclbin/${{KERNEL}}.${{TARGET}}.xclbin"
if [ ! -f "$XCLBIN_FILE" ]; then
    echo "Error: XCLBIN file not found at '$XCLBIN_FILE'"
    echo "Please build for the target '$TARGET' first using: make all TARGET=$TARGET"
    exit 1
fi

# 3. Run Host Executable
DATASET="./graph.txt"
./${{EXECUTABLE}} ${{XCLBIN_FILE}} $DATASET
"""
        (files["run.sh"])  # Make it executable - this is a comment, actual chmod happens in project_generator

        # system.cfg
        files[
            "system.cfg"
        ] = f"""
# This file is auto-generated by GraphyFlow.
[connectivity]
nk={kernel_name}:1:{kernel_name}_1
"""
        # scripts/kernel/kernel.mk
        files[
            "scripts/kernel/kernel.mk"
        ] = f"""
# Makefile for the Vitis Kernel
VPP := v++
# KERNEL_NAME is passed from the top Makefile
KERNEL_SRC := scripts/kernel/$(KERNEL_NAME).cpp
XCLBIN_DIR := ./xclbin
XCLBIN_FILE := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xclbin
KERNEL_XO := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xo
EMCONFIG_FILE := ./emconfig.json

# Include host directory for common.h
CLFLAGS += --kernel $(KERNEL_NAME) -Iscripts/kernel -Iscripts/host

LDFLAGS_VPP += --config ./system.cfg

$(KERNEL_XO): $(KERNEL_SRC)
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) $(CLFLAGS) -o $@ $<

$(XCLBIN_FILE): $(KERNEL_XO)
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) $(LDFLAGS_VPP) -o $@ $<

emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig
"""
        return files

    def debug_msgs(self, phases=[1, 2, 3]):
        if 1 in phases:
            # For demonstration, print discovered types
            print("--- Discovered Struct Definitions ---")
            for name, (hls_type, members) in self.struct_definitions.items():
                print(f"Struct: {name}")
                print(hls_type.gen_decl(members))

            print("\n--- Discovered Batch Type Mappings ---")
            for base, batch in self.batch_type_map.items():
                print(f"Base Type: {base.name} -> Batch Type: {batch.name}")
        if 2 in phases:
            # For demonstration, print discovered functions and streams
            print("\n--- Discovered HLS Functions and Signatures ---")
            for func in self.hls_functions.values():
                param_str = ", ".join([f"{p.type.name}& {p.name}" for p in func.params])
                stream_status = "Streamed" if func.streamed else "Unstreamed (by-ref)"
                print(f"Function: {func.name} ({stream_status})")
                print(f"  Signature: void {func.name}({param_str});")

            print("\n--- Intermediate Streams for Top-Level Function ---")
            for decl in self.top_level_stream_decls:
                print(decl.gen_code(indent_lvl=1).strip())
        if 3 in phases:
            print("\n--- Generated HLS Code Bodies (Phase 3) ---")
            for func in self.hls_functions.values():
                print(f"// ======== Code for function: {func.name} ========")
                for code_line in func.codes:
                    # The gen_code method of each HLSCodeLine object produces the C++ string
                    print(code_line.gen_code(indent_lvl=1), end="")
                print(f"// ======== End of function: {func.name} ========\n")

    def _find_unstreamed_funcs(self, comp_col: dfir.ComponentCollection):
        """
        Identifies all components that are part of a ReduceComponent's sub-graph.
        Their function signatures will use pass-by-reference instead of streams.
        """
        self.unstreamed_funcs.clear()
        q = []
        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                for port_name in [
                    "o_reduce_key_in",
                    "o_reduce_transform_in",
                    "o_reduce_unit_start_0",
                    "o_reduce_unit_start_1",
                ]:
                    if comp.get_port(port_name).connected:
                        q.append(comp.get_port(port_name).connection.parent)

        visited = set()
        while q:
            comp: dfir.Component = q.pop(0)
            if comp.readable_id in visited:
                continue
            visited.add(comp.readable_id)

            if isinstance(comp, dfir.ReduceComponent):
                continue

            self.unstreamed_funcs.add(f"{comp.__class__.__name__[:5]}_{comp.readable_id}")
            for port in comp.out_ports:
                if port.connected:
                    q.append(port.connection.parent)

    def _analyze_and_map_types(self, comp_col: dfir.ComponentCollection):
        """
        Phase 1: Traverse all components and ports to analyze and map DFIR types
        to HLS types, including special batching types for streams.
        """
        self.type_map.clear()
        self.batch_type_map.clear()
        self.struct_definitions.clear()

        # First, identify all functions that are part of a reduce operation
        self._find_unstreamed_funcs(comp_col)

        # Iterate all ports of all components to discover all necessary types
        for comp in comp_col.components:
            for port in comp.ports:
                dfir_type = port.data_type
                is_array_type = False
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                    is_array_type = True

                if dfir_type:
                    # Get the base HLS type (e.g., a struct without batching wrappers)
                    base_hls_type = self._to_hls_type(dfir_type, is_array_type)

                    # If it's a stream port (default case) and not for a reduce sub-function,
                    # create a corresponding batch type.
                    is_stream_port = comp.name not in self.unstreamed_funcs
                    if is_stream_port:
                        self._get_batch_type(base_hls_type)

    def generate_common_header(self, top_func_name: str) -> str:
        """
        Generates the full content of the common.h header file.
        This version now INCLUDES the GraphCSR definition.
        """
        # 确保 host output types 被定义并注册
        self._get_host_output_type()

        header_guard = "__COMMON_H__"
        code = f"#ifndef {header_guard}\n#define {header_guard}\n\n"

        code += "#include <limits>\n"
        code += "#include <string>\n"
        code += "#include <vector>\n\n"
        code += '#ifndef __SYNTHESIS__\n#include "xcl2.h"\n#endif\n\n'

        code += "// A constant representing infinity for distance initialization\n"
        code += "const int INFINITY_DIST = 16384;\n\n"
        code += "// Structure to hold the graph in Compressed Sparse Row (CSR) format\n"
        code += "struct GraphCSR {\n"
        code += "    int num_vertices;\n"
        code += "    int num_edges;\n"
        code += "    std::vector<int> offsets;\n"
        code += "    std::vector<int> columns;\n"
        code += "    std::vector<int> weights;\n"
        code += "};\n\n"

        code += "#include <ap_fixed.h>\n#include <stdint.h>\n\n"
        code += f"#define PE_NUM {self.PE_NUM}\n\n"

        code += "// --- Struct Type Definitions ---\n"
        sorted_defs = self._topologically_sort_structs()
        for hls_type, members in sorted_defs:
            code += hls_type.gen_decl(members) + "\n"

        code += f"#endif // {header_guard}\n"
        return code

    def _define_functions_and_streams(self, comp_col: dfir.ComponentCollection, top_func_name: str):
        """
        Phase 2: Creates HLSFunction objects, defines their signatures, and
        identifies the intermediate streams needed for the top-level function.
        """
        self.hls_functions.clear()
        self.top_level_stream_decls.clear()
        self.reduce_helpers.clear()
        self.utility_functions.clear()

        processed_sub_comp_ids = set()

        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                pre_process_func = HLSFunction(name=f"{comp.name}_pre_process", comp=comp)
                unit_reduce_func = HLSFunction(name=f"{comp.name}_unit_reduce", comp=comp)

                in_type = self.type_map[comp.get_port("i_0").data_type]
                key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
                transform_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]

                pre_process_func.params = [
                    HLSVar("i_0", HLSType(HLSBasicType.STREAM, [self.batch_type_map[in_type]])),
                    HLSVar("intermediate_key", HLSType(HLSBasicType.STREAM, [self.batch_type_map[key_type]])),
                    HLSVar(
                        "intermediate_transform",
                        HLSType(HLSBasicType.STREAM, [self.batch_type_map[transform_type]]),
                    ),
                ]

                kt_pair_type = HLSType(
                    HLSBasicType.STRUCT,
                    [key_type, transform_type],
                    struct_name=f"kt_pair_{comp.readable_id}_t",
                    struct_prop_names=["key", "transform"],
                )
                self.struct_definitions[kt_pair_type.name] = (kt_pair_type, ["key", "transform"])
                net_wrapper_type = HLSType(
                    HLSBasicType.STRUCT,
                    [kt_pair_type, HLSType(HLSBasicType.BOOL)],
                    struct_name=f"net_wrapper_{kt_pair_type.name}_t",
                    struct_prop_names=["data", "end_flag"],
                )
                self.struct_definitions[net_wrapper_type.name] = (net_wrapper_type, ["data", "end_flag"])

                out_dfir_type = comp.get_port("o_0").data_type
                base_out_batch_type = self._get_batch_type(self.type_map[out_dfir_type])
                unit_reduce_func.params = [
                    HLSVar(
                        "kt_wrap_item",
                        HLSType(
                            HLSBasicType.ARRAY,
                            [HLSType(HLSBasicType.STREAM, [net_wrapper_type])],
                            array_dims=["PE_NUM"],
                        ),
                    ),
                    HLSVar("o_0", HLSType(HLSBasicType.STREAM, [base_out_batch_type])),
                ]

                self.hls_functions[pre_process_func.readable_id] = pre_process_func
                self.hls_functions[unit_reduce_func.readable_id] = unit_reduce_func

                key_batch_type = self.batch_type_map[key_type]
                transform_batch_type = self.batch_type_map[transform_type]
                kt_pair_batch_type = self._get_batch_type(kt_pair_type)
                zipper_func = generate_stream_zipper(key_batch_type, transform_batch_type, kt_pair_batch_type)
                demux_func = generate_demux(self.PE_NUM, kt_pair_batch_type, net_wrapper_type)
                omega_funcs = generate_omega_network(self.PE_NUM, net_wrapper_type, routing_key_member="key")
                omega_func = next(f for f in omega_funcs if "omega_switch" in f.name)
                self.utility_functions.extend([zipper_func, demux_func] + omega_funcs)

                helpers = {
                    "pre_process": pre_process_func,
                    "unit_reduce": unit_reduce_func,
                    "zipper": zipper_func,
                    "demux": demux_func,
                    "omega": omega_func,
                }

                # --- *** 关键修正：移除多余的 internal_streams 逻辑，只保留 streams_to_declare *** ---
                streams_to_declare = {
                    "zipper_to_demux": HLSVar(
                        f"reduce_{comp.readable_id}_z2d_pair",
                        HLSType(HLSBasicType.STREAM, [kt_pair_batch_type]),
                    ),
                    "demux_to_omega": HLSVar(
                        f"reduce_{comp.readable_id}_d2o_pair", demux_func.params[1].type
                    ),
                    "omega_to_unit": HLSVar(f"reduce_{comp.readable_id}_o2u_pair", omega_func.params[1].type),
                    # 从 pre_process_func.params 直接获取正确的 HLSVar 对象
                    "intermediate_key": pre_process_func.params[1],
                    "intermediate_transform": pre_process_func.params[2],
                }

                for stream_var in streams_to_declare.values():
                    # 为顶层函数声明这些流
                    decl = CodeVarDecl(stream_var.name, stream_var.type)
                    pragma = CodePragma(f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}")
                    self.top_level_stream_decls.append((decl, pragma))

                helpers["streams"] = streams_to_declare
                self.reduce_helpers[comp.readable_id] = helpers

                # 标记子图组件为已处理 (逻辑不变)
                for port_name in [
                    "o_reduce_key_in",
                    "o_reduce_transform_in",
                    "o_reduce_unit_start_0",
                    "o_reduce_unit_start_1",
                ]:
                    if comp.get_port(port_name).connected:
                        q = [comp.get_port(port_name).connection.parent]
                        visited_sub = set()
                        while q:
                            sub_comp = q.pop(0)
                            if sub_comp.readable_id in visited_sub:
                                continue
                            visited_sub.add(sub_comp.readable_id)
                            processed_sub_comp_ids.add(sub_comp.readable_id)
                            for p in sub_comp.out_ports:
                                if p.connected and not isinstance(p.connection.parent, dfir.ReduceComponent):
                                    q.append(p.connection.parent)

        # 处理普通组件 (逻辑不变)
        for comp in comp_col.components:
            if comp.readable_id in processed_sub_comp_ids or isinstance(
                comp,
                (
                    dfir.IOComponent,
                    dfir.ConstantComponent,
                    dfir.UnusedEndMarkerComponent,
                    dfir.ReduceComponent,
                ),
            ):
                continue
            hls_func = HLSFunction(name=comp.name, comp=comp)
            for port in comp.ports:
                if port.connection and isinstance(
                    port.connection.parent, (dfir.UnusedEndMarkerComponent, dfir.ConstantComponent)
                ):
                    continue
                dfir_type = (
                    port.data_type.type_ if isinstance(port.data_type, dftype.ArrayType) else port.data_type
                )
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                param_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                hls_func.params.append(HLSVar(var_name=port.name, var_type=param_type))
            self.hls_functions[comp.readable_id] = hls_func

        # 声明中间流 (逻辑不变)
        all_stream_comp_ids = {f.dfir_comp.readable_id for f in self.hls_functions.values()}
        visited_ports = set()
        for port in comp_col.all_connected_ports:
            if port.readable_id in visited_ports:
                continue
            conn = port.connection
            is_intermediate = (
                port.parent.readable_id in all_stream_comp_ids
                and conn.parent.readable_id in all_stream_comp_ids
            )
            if is_intermediate:
                dfir_type = (
                    port.data_type.type_ if isinstance(port.data_type, dftype.ArrayType) else port.data_type
                )
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                stream_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                out_port = port if port.port_type == dfir.PortType.OUT else conn
                stream_name = f"stream_{out_port.unique_name}"
                decl = CodeVarDecl(stream_name, stream_type)
                pragma = CodePragma(f"STREAM variable={stream_name} depth={self.STREAM_DEPTH}")
                self.top_level_stream_decls.append((decl, pragma))
            visited_ports.add(port.readable_id)
            visited_ports.add(conn.readable_id)

    def _translate_functions(self):
        """Phase 3 Entry Point: Populates the .codes for all HLSFunctions."""
        # --- MODIFIED: Handle ReduceComponent first ---
        reduce_comps = [
            f.dfir_comp for f in self.hls_functions.values() if isinstance(f.dfir_comp, dfir.ReduceComponent)
        ]
        for comp in reduce_comps:
            pre_process_func = next(
                f for f in self.hls_functions.values() if f.name == f"{comp.name}_pre_process"
            )
            unit_reduce_func = next(
                f for f in self.hls_functions.values() if f.name == f"{comp.name}_unit_reduce"
            )

            pre_process_func.codes = self._translate_reduce_preprocess(pre_process_func)
            unit_reduce_func.codes = self._translate_reduce_unit_reduce(unit_reduce_func)

        # --- Translate other functions ---
        for func in self.hls_functions.values():
            if not isinstance(func.dfir_comp, dfir.ReduceComponent):
                if func.streamed:
                    self._translate_streamed_component(func)
                else:  # Should not happen with the new logic
                    assert False
                    self._translate_unstreamed_component(func)

    def _generate_mem_to_stream_func(self) -> HLSFunction:
        """Generates the function that reads from AXI pointers into streams."""
        func = HLSFunction("mem_to_stream_func", comp=None)

        params = []
        body = []

        for port in self.axi_input_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]

            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[batch_type], is_const_ptr=True)
            axi_param = HLSVar(f"in_{port.unique_name}", axi_ptr_type)
            params.append(axi_param)

            stream_type = HLSType(HLSBasicType.STREAM, [batch_type])
            stream_param = HLSVar(f"out_{port.unique_name}_stream", stream_type)
            params.append(stream_param)

            loop = CodeFor(
                codes=[
                    CodePragma("PIPELINE"),
                    CodeWriteStream(
                        stream_param, HLSExpr(HLSExprT.VAR, HLSVar(f"{axi_param.name}[i]", batch_type))
                    ),
                ],
                iter_limit="num_batches",
                iter_name="i",
            )
            body.append(loop)

        num_batches_type = HLSType(HLSBasicType.UINT16)
        params.append(HLSVar("num_batches", num_batches_type))

        func.params = params
        func.codes = body
        return func

    def _generate_stream_to_mem_func(self) -> HLSFunction:
        """Generates the function that writes from streams to AXI pointers."""
        func = HLSFunction("stream_to_mem_func", comp=None)

        params = []
        body = []

        for port in self.axi_output_ports:
            internal_batch_type = self.batch_type_map[self.type_map[port.data_type]]
            stream_param = HLSVar(
                f"in_{port.unique_name}_stream", HLSType(HLSBasicType.STREAM, [internal_batch_type])
            )
            params.append(stream_param)
            host_output_type = self._get_host_output_type()
            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[host_output_type], is_const_ptr=False)
            axi_param = HLSVar(f"out_{port.unique_name}", axi_ptr_type)
            params.append(axi_param)

            i_var = HLSVar("i", HLSType(HLSBasicType.INT))
            internal_batch_var = HLSVar("internal_batch", internal_batch_type)
            output_batch_var = HLSVar("output_batch", host_output_type)

            # New variables for casting
            final_dist_fp_var = HLSVar("final_dist_fp", HLSType(HLSBasicType.FLOAT))
            internal_ele_0_var = HLSVar(f"internal_batch.data[k].ele_0", HLSType(HLSBasicType.AP_FIXED_POD))

            conversion_loop = CodeFor(
                [
                    CodePragma("UNROLL"),
                    CodeVarDecl(final_dist_fp_var.name, final_dist_fp_var.type),
                    CodeAssign(
                        final_dist_fp_var,
                        HLSExpr(
                            HLSExprT.VAR,
                            HLSVar(
                                f"*reinterpret_cast<ap_fixed<32, 16>*>(&{internal_ele_0_var.name})",
                                HLSType(HLSBasicType.FLOAT),
                            ),
                        ),
                    ),
                    CodeAssign(
                        HLSVar(
                            f"{output_batch_var.name}.data[k].distance",
                            HLSType(HLSBasicType.REAL_FLOAT),
                        ),
                        HLSExpr(
                            HLSExprT.VAR,
                            HLSVar(f"(float){final_dist_fp_var.name}", HLSType(HLSBasicType.REAL_FLOAT)),
                        ),
                    ),
                    CodeAssign(
                        HLSVar(f"{output_batch_var.name}.data[k].id", HLSType(HLSBasicType.INT)),
                        HLSExpr(
                            HLSExprT.VAR,
                            HLSVar(f"{internal_batch_var.name}.data[k].ele_1.id", HLSType(HLSBasicType.INT)),
                        ),
                    ),
                ],
                "PE_NUM",
                iter_name="k",
            )

            while_body = [
                CodePragma("PIPELINE"),
                CodeVarDecl(internal_batch_var.name, internal_batch_var.type),
                CodeAssign(
                    internal_batch_var,
                    HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, stream_param)]),
                ),
                CodeVarDecl(output_batch_var.name, output_batch_var.type),
                conversion_loop,
                CodeAssign(
                    HLSVar(f"{output_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                    HLSExpr(
                        HLSExprT.UOP,
                        (UnaryOp.GET_ATTR, "end_flag"),
                        [HLSExpr(HLSExprT.VAR, internal_batch_var)],
                    ),
                ),
                CodeAssign(
                    HLSVar(f"{output_batch_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                    HLSExpr(
                        HLSExprT.UOP,
                        (UnaryOp.GET_ATTR, "end_pos"),
                        [HLSExpr(HLSExprT.VAR, internal_batch_var)],
                    ),
                ),
                CodeAssign(
                    HLSVar(f"{axi_param.name}[i]", host_output_type), HLSExpr(HLSExprT.VAR, output_batch_var)
                ),
                CodeIf(
                    HLSExpr(
                        HLSExprT.VAR, HLSVar(f"{axi_param.name}[i].end_flag", HLSType(HLSBasicType.BOOL))
                    ),
                    [CodeBreak()],
                ),
                CodeAssign(
                    i_var,
                    HLSExpr(
                        HLSExprT.BINOP, BinOp.ADD, [HLSExpr(HLSExprT.VAR, i_var), HLSExpr(HLSExprT.CONST, 1)]
                    ),
                ),
            ]

            body.extend(
                [
                    CodeVarDecl(i_var.name, i_var.type, init_val=0),
                    CodeWhile(while_body, HLSExpr(HLSExprT.CONST, True)),
                ]
            )

        func.params = params
        func.codes = body
        return func

    def _generate_dataflow_core_func(self, top_func_name: str) -> HLSFunction:
        """Phase 4.3: Generates the implementation of the core dataflow function (the old top-level)."""
        # This function is largely the same as the old _generate_top_level_function
        # but with a new name and a signature composed of only streams.
        func = HLSFunction(f"{top_func_name}_dataflow", comp=None)

        params = []
        for port in self.axi_input_ports + self.axi_output_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]
            stream_type = HLSType(HLSBasicType.STREAM, [batch_type])
            params.append(HLSVar(f"{port.unique_name}_stream", stream_type))
        func.params = params

        # The body generation is the same as the original _generate_top_level_function
        func.codes = self._generate_top_level_function_body()  # Delegate body generation
        return func

    def _generate_axi_kernel_wrapper(self, top_func_name: str) -> Tuple[str, str]:
        """Generates the final extern "C" kernel with AXI pragmas placed correctly."""
        params_str_list = []
        param_vars = []

        # 1. Build parameter list for the top-level function signature
        for port in self.axi_input_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]
            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[batch_type], is_const_ptr=True)
            axi_param = HLSVar(port.unique_name, axi_ptr_type)
            params_str_list.append(axi_param.type.get_upper_decl(axi_param.name))
            param_vars.append(axi_param)

        for port in self.axi_output_ports:
            host_output_type = self._get_host_output_type()
            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[host_output_type], is_const_ptr=False)
            axi_param = HLSVar(port.unique_name, axi_ptr_type)
            params_str_list.append(axi_param.type.get_upper_decl(axi_param.name))
            param_vars.append(axi_param)

        params_str_list.append("int* stop_flag")
        param_vars.append(HLSVar("stop_flag", HLSType(HLSBasicType.POINTER, [HLSType(HLSBasicType.INT)])))
        params_str_list.append("uint16_t input_length_in_batches")
        param_vars.append(HLSVar("input_length_in_batches", HLSType(HLSBasicType.UINT16)))

        top_func_sig = (
            f'extern "C" void {top_func_name}(\n{INDENT_UNIT}'
            + f",\n{INDENT_UNIT}".join(params_str_list)
            + "\n)"
        )

        # 2. Prepare Pragmas and Body
        pragmas = []
        for i, var in enumerate(param_vars):
            bundle = f"gmem{i}"
            if var.type.type == HLSBasicType.POINTER:
                pragmas.append(f"#pragma HLS INTERFACE m_axi port={var.name} offset=slave bundle={bundle}")
        for var in param_vars:
            pragmas.append(f"#pragma HLS INTERFACE s_axilite port={var.name}")
        pragmas.append("#pragma HLS INTERFACE s_axilite port=return")

        body: List[HLSCodeLine] = []

        # --- *** 关键修正：将 Pragma 作为 CodeLine 对象添加到函数体列表的开头 *** ---
        for p_str in pragmas:
            body.append(CodePragma(p_str.replace("#pragma HLS ", "")))

        internal_streams = []
        for port in self.top_level_io_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]
            stream_var = HLSVar(
                f"{port.unique_name}_internal_stream", HLSType(HLSBasicType.STREAM, [batch_type])
            )
            internal_streams.append(stream_var)
            decl = CodeVarDecl(stream_var.name, stream_var.type)
            setattr(decl, "is_static", True)
            body.append(decl)
            body.append(CodePragma(f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}"))

        body.append(CodePragma("DATAFLOW"))

        # Generate calls (logic remains the same)
        m2s_axi_params = [p for p in param_vars if p.type.is_const_ptr]
        m2s_stream_params = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_input_ports)
        ]
        m2s_len_param = [p for p in param_vars if "input_length" in p.name]
        body.append(CodeCall(self.mem_to_stream_func, m2s_axi_params + m2s_stream_params + m2s_len_param))

        df_core_input_streams = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_input_ports)
        ]
        df_core_output_streams = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_output_ports)
        ]
        body.append(CodeCall(self.dataflow_core_func, df_core_input_streams + df_core_output_streams))

        s2m_stream_params = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_output_ports)
        ]
        s2m_axi_params = [
            p
            for p in param_vars
            if p.type.type == HLSBasicType.POINTER and not p.type.is_const_ptr and "stop_flag" not in p.name
        ]
        body.append(CodeCall(self.stream_to_mem_func, s2m_stream_params + s2m_axi_params))

        # 3. Assemble final C++ string
        code = f"{top_func_sig} " + "{\n"
        for line in body:
            if isinstance(line, CodeVarDecl) and getattr(line, "is_static", False):
                code += f"{INDENT_UNIT}static {line.gen_code().lstrip()}"
            else:
                code += line.gen_code(1)
        code += "}\n"

        return code, top_func_sig

    # ======================================================================== #
    #                            PHASE 1                                       #
    # ======================================================================== #

    def _get_batch_type(self, base_type: HLSType) -> HLSType:
        """
        Creates (or retrieves from cache) a batched version of a base HLSType.
        The batched type is a struct containing an array of the base type and control flags.
        """
        if base_type in self.batch_type_map:
            return self.batch_type_map[base_type]

        data_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[base_type], array_dims=["PE_NUM"])
        end_flag_type = HLSType(HLSBasicType.BOOL)
        end_pos_type = HLSType(HLSBasicType.UINT8)

        member_types = [data_array_type, end_flag_type, end_pos_type]
        member_names = ["data", "end_flag", "end_pos"]

        batch_type = HLSType(HLSBasicType.STRUCT, member_types, struct_prop_names=member_names)

        self.batch_type_map[base_type] = batch_type
        if batch_type.name not in self.struct_definitions:
            self.struct_definitions[batch_type.name] = (batch_type, member_names)

        return batch_type

    def _to_hls_type(self, dfir_type: dftype.DfirType, is_array_type: bool = False) -> HLSType:
        """
        Recursively converts a DfirType to a base HLSType, using memoization.
        This handles basic types, tuples, optionals, and special graph types.
        """
        global_graph = self.global_graph_store
        if dfir_type in self.type_map:
            if is_array_type and dfir.ArrayType(dfir_type) not in self.type_map:
                self.type_map[dfir.ArrayType(dfir_type)] = self.type_map[dfir_type]
            return self.type_map[dfir_type]

        hls_type: HLSType

        if isinstance(dfir_type, dftype.IntType):
            hls_type = HLSType(HLSBasicType.INT)
        elif isinstance(dfir_type, dftype.FloatType):
            hls_type = HLSType(HLSBasicType.AP_FIXED_POD)  # MODIFIED
        elif isinstance(dfir_type, dftype.BoolType):
            hls_type = HLSType(HLSBasicType.BOOL)
        elif isinstance(dfir_type, dftype.TupleType):
            sub_types = [self._to_hls_type(t) for t in dfir_type.types]
            member_names = [f"ele_{i}" for i in range(len(sub_types))]
            hls_type = HLSType(HLSBasicType.STRUCT, sub_types, struct_prop_names=member_names)
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, member_names)
        elif isinstance(dfir_type, dftype.OptionalType):
            data_type = self._to_hls_type(dfir_type.type_)
            valid_type = HLSType(HLSBasicType.BOOL)
            hls_type = HLSType(
                HLSBasicType.STRUCT,
                sub_types=[data_type, valid_type],
                struct_name=f"opt_{data_type.name}_t",
                struct_prop_names=["data", "valid"],
            )
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, ["data", "valid"])
        elif isinstance(dfir_type, dftype.SpecialType):
            props = (
                global_graph.node_properties
                if dfir_type.type_name == "node"
                else global_graph.edge_properties
            )
            prop_names = list(props.keys())
            prop_types = [self._to_hls_type(t) for t in props.values()]
            struct_name = f"{dfir_type.type_name}_t"
            hls_type = HLSType(HLSBasicType.STRUCT, prop_types, struct_name, prop_names)
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, prop_names)
        else:
            raise NotImplementedError(f"DFIR type conversion not implemented for {type(dfir_type)}")

        self.type_map[dfir_type] = hls_type
        if is_array_type:
            self.type_map[dfir.ArrayType(dfir_type)] = hls_type
        return hls_type

    # ======================================================================== #
    #                            PHASE 3                                       #
    # ======================================================================== #

    def _translate_streamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component into a standard streamed HLS function body."""
        comp = hls_func.dfir_comp

        # Dispatcher to select the correct translation logic
        if isinstance(comp, dfir.BinOpComponent):
            inner_logic = self._translate_binop_op(comp, "i")
        elif isinstance(comp, dfir.UnaryOpComponent):
            inner_logic = self._translate_unary_op(comp, "i")
        elif isinstance(comp, dfir.CopyComponent):
            inner_logic = self._translate_copy_op(comp, "i")
        elif isinstance(comp, dfir.GatherComponent):
            inner_logic = self._translate_gather_op(comp, "i")
        elif isinstance(comp, dfir.ScatterComponent):
            inner_logic = self._translate_scatter_op(comp, "i")
        elif isinstance(comp, dfir.ConditionalComponent):
            inner_logic = self._translate_conditional_op(comp, "i")
        elif isinstance(comp, dfir.CollectComponent):
            # Collect has a different boilerplate, handle it separately
            hls_func.codes = self._translate_collect_op(hls_func)
            return
        else:
            inner_logic = [
                CodePragma(f"WARNING: Component {type(comp).__name__} translation not implemented.")
            ]

        # Wrap the core logic in the standard streaming boilerplate
        hls_func.codes = self._generate_streamed_function_boilerplate(hls_func, inner_logic)

    def _translate_unstreamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component for an unstreamed (pass-by-reference) function."""
        # This is a placeholder for now, as it's mainly for reduce sub-functions (key, transform, unit)
        hls_func.codes = [
            CodePragma("INLINE"),
            CodePragma(f"WARNING: Unstreamed func translation not fully implemented for {hls_func.name}"),
        ]

    def _generate_streamed_function_boilerplate(
        self, hls_func: HLSFunction, inner_loop_logic: List[HLSCodeLine]
    ) -> List[HLSCodeLine]:
        """Creates the standard while/for loop structure for a streamed function."""
        body: List[HLSCodeLine] = []
        in_ports = hls_func.dfir_comp.in_ports.copy()
        out_ports = hls_func.dfir_comp.out_ports.copy()

        # 1. Declare local batch variables for inputs and outputs
        in_batch_vars: Dict[str, HLSVar] = {
            p.name: HLSVar(f"in_batch_{p.name}", p.type.sub_types[0])
            for p in hls_func.params
            if p.name in [ip.name for ip in in_ports]
        }
        out_batch_vars: Dict[str, HLSVar] = {
            p.name: HLSVar(f"out_batch_{p.name}", p.type.sub_types[0])
            for p in hls_func.params
            if p.name in [op.name for op in out_ports]
        }
        for var in list(in_batch_vars.values()) + list(out_batch_vars.values()):
            body.append(CodeVarDecl(var.name, var.type))
        end_flag_var_decl = CodeVarDecl("end_flag", HLSType(HLSBasicType.BOOL))
        end_flag_var = end_flag_var_decl.var
        body.append(end_flag_var_decl)
        end_pos_var_decl = CodeVarDecl("end_pos", HLSType(HLSBasicType.UINT8))
        end_pos_var = end_pos_var_decl.var
        body.append(end_pos_var_decl)

        # 2. Create the main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 3. Read from all input streams
        for p in hls_func.params:
            if p.name in in_batch_vars:
                read_expr = HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, p)])
                while_loop_body.append(CodeAssign(in_batch_vars[p.name], read_expr))

        # 4. Create the inner for loop
        for_loop = CodeFor(
            codes=[CodePragma("UNROLL")] + inner_loop_logic,
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(for_loop)

        # 5. Get end flag value.
        assert in_batch_vars
        # Combine end flags from all inputs
        # For simplicity, we use the first input's end_flag. A real implementation might OR them.
        first_in_batch = list(in_batch_vars.values())[0]
        end_check_expr = HLSExpr(
            HLSExprT.VAR,
            HLSVar(f"{first_in_batch.name}.end_flag", end_flag_var.type),
        )
        assign_end_flag = CodeAssign(end_flag_var, end_check_expr)
        end_check_pos_expr = HLSExpr(
            HLSExprT.VAR,
            HLSVar(f"{first_in_batch.name}.end_pos", end_pos_var.type),
        )
        assign_end_pos = CodeAssign(end_pos_var, end_check_pos_expr)
        while_loop_body.extend([assign_end_flag, assign_end_pos])

        # 6. Write to all output streams
        for p in hls_func.params:
            if p.name in out_batch_vars:
                # assign end_flag & end pos
                while_loop_body.append(
                    CodeAssign(
                        HLSVar(f"{out_batch_vars[p.name].name}.end_flag", end_flag_var.type),
                        HLSExpr(HLSExprT.VAR, end_flag_var),
                    )
                )
                while_loop_body.append(
                    CodeAssign(
                        HLSVar(f"{out_batch_vars[p.name].name}.end_pos", end_pos_var.type),
                        HLSExpr(HLSExprT.VAR, end_pos_var),
                    )
                )
                while_loop_body.append(CodeWriteStream(p, out_batch_vars[p.name]))

        # 7. Check for end condition and break
        if in_batch_vars:
            break_if = CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()])
            while_loop_body.extend([break_if])

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))
        return body

    def _add_pod_to_float_cast(
        self, pod_expr: HLSExpr, code_list: List[HLSCodeLine], base_name: str
    ) -> HLSExpr:
        """
        Generates code to cast a POD type (int32_t) to a computational float (ap_fixed).
        Handles both variables and constants correctly by returning an HLSExpr.
        Appends prerequisite declarations to code_list for variables.
        """
        # If the expression is a constant, return a direct C++ cast expression.
        # This will generate "((ap_fixed<32, 16>)0.0)" which is legal C++.
        if pod_expr.type == HLSExprT.CONST:
            return HLSExpr(HLSExprT.UOP, UnaryOp.CAST_FLOAT, [pod_expr])

        # If the expression is a variable, generate the reinterpret_cast logic.
        elif pod_expr.type == HLSExprT.VAR:
            ap_fixed_type = HLSType(HLSBasicType.FLOAT)
            float_var = HLSVar(base_name, ap_fixed_type)

            # 1. Declare a new ap_fixed variable.
            # 2. Initialize it by reinterpreting the bits of the input POD variable.
            cast_str = f"*reinterpret_cast<ap_fixed<32, 16>*>(&{pod_expr.code})"
            code_list.append(CodeVarDecl(float_var.name, float_var.type, init_val=cast_str))

            # 3. Return an expression that refers to this new temporary variable.
            return HLSExpr(HLSExprT.VAR, float_var)

        else:
            raise TypeError(f"Unsupported HLSExpr type for casting: {pod_expr.type}")

    def _add_float_to_pod_cast(self, float_var: HLSVar, code_list: List[HLSCodeLine], target_pod_var: HLSVar):
        """
        Generates code to cast a computational float (ap_fixed) back to a POD type (int32_t).
        (This function remains unchanged as its input is always a variable)
        """
        cast_str = f"*reinterpret_cast<int32_t*>(&{float_var.name})"
        assign_expr = HLSExpr(HLSExprT.VAR, HLSVar(cast_str, target_pod_var.type))
        code_list.append(CodeAssign(target_pod_var, assign_expr))

    # --- Component-Specific Translators for Inner Loop Logic ---

    # In graphyflow/backend_manager.py, replace the existing _translate_binop_op function

    def _translate_binop_op(self, comp: dfir.BinOpComponent, iterator: str) -> List[HLSCodeLine]:
        """
        Generates the core logic for a BinOpComponent.
        This version uses the robust _add_pod_to_float_cast helper.
        """
        in0_type = self.batch_type_map[self.type_map[comp.input_type]].sub_types[0].sub_types[0]
        out_type = self.batch_type_map[self.type_map[comp.output_type]].sub_types[0].sub_types[0]

        op1_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in0_type))
        op1_expr = HLSExpr.check_const(op1_expr, comp.in_ports[0])
        op2_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_1.data[{iterator}]", in0_type))
        op2_expr = HLSExpr.check_const(op2_expr, comp.in_ports[1])
        target_var = HLSVar(f"out_batch_o_0.data[{iterator}]", out_type)

        if in0_type.type == HLSBasicType.AP_FIXED_POD:
            code = []
            is_comparison = comp.op in [BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.GT, BinOp.LE, BinOp.GE]

            # Unified handling for both constants and variables
            final_op1_expr = self._add_pod_to_float_cast(op1_expr, code, f"val1_{comp.readable_id}")
            final_op2_expr = self._add_pod_to_float_cast(op2_expr, code, f"val2_{comp.readable_id}")

            bin_expr = HLSExpr(HLSExprT.BINOP, comp.op, [final_op1_expr, final_op2_expr])

            if is_comparison:
                code.append(CodeAssign(target_var, bin_expr))
            else:  # Arithmetic or MIN/MAX
                ap_fixed_type = HLSType(HLSBasicType.FLOAT)
                result_var = HLSVar(f"result_{comp.readable_id}", ap_fixed_type)
                code.append(CodeVarDecl(result_var.name, result_var.type))
                code.append(CodeAssign(result_var, bin_expr))
                self._add_float_to_pod_cast(result_var, code, target_var)

            return code
        else:
            bin_expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1_expr, op2_expr])
            return [CodeAssign(target_var, bin_expr)]

    def _translate_unary_op(self, comp: dfir.UnaryOpComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a UnaryOpComponent."""
        in_type = self.batch_type_map[self.type_map[comp.input_type]].sub_types[0].sub_types[0]
        out_type = self.batch_type_map[self.type_map[comp.output_type]].sub_types[0].sub_types[0]

        operand = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in_type))
        operand = HLSExpr.check_const(operand, comp.in_ports[0])
        comp_op_var = comp.op
        if comp.op == UnaryOp.GET_ATTR:
            assert operand.val.type.type == HLSBasicType.STRUCT
            comp_op_var = (comp_op_var, comp.select_index)
        unary_expr = HLSExpr(HLSExprT.UOP, comp_op_var, [operand])
        target_var = HLSVar(f"out_batch_o_0.data[{iterator}]", out_type)

        return [CodeAssign(target_var, unary_expr)]

    def _translate_copy_op(self, comp: dfir.CopyComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a CopyComponent."""
        in_type = self.type_map[comp.get_port("i_0").data_type]

        in_var_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in_type))
        in_var_expr = HLSExpr.check_const(in_var_expr, comp.in_ports[0])

        target_o0 = HLSVar(f"out_batch_o_0.data[{iterator}]", in_type)
        target_o1 = HLSVar(f"out_batch_o_1.data[{iterator}]", in_type)

        return [CodeAssign(target_o0, in_var_expr), CodeAssign(target_o1, in_var_expr)]

    def _translate_gather_op(self, comp: dfir.GatherComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a GatherComponent."""
        out_port = comp.get_port("o_0")
        out_type = self.type_map[out_port.data_type]

        assignments = []
        for i, in_port in enumerate(comp.in_ports):
            in_type = self.type_map[in_port.data_type]
            in_var_expr = HLSExpr(
                HLSExprT.VAR,
                HLSVar(f"in_batch_{in_port.name}.data[{iterator}]", in_type),
            )
            in_var_expr = HLSExpr.check_const(in_var_expr, comp.in_ports[i])

            # Target is a member of the output struct
            target_member = HLSVar(f"out_batch_o_0.data[{iterator}].ele_{i}", in_type)
            assignments.append(CodeAssign(target_member, in_var_expr))

        return assignments

    def _translate_scatter_op(self, comp: dfir.ScatterComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a ScatterComponent."""
        in_port = comp.get_port("i_0")
        in_type = self.type_map[in_port.data_type]

        assignments = []
        for i, out_port in enumerate(comp.out_ports):
            if isinstance(out_port.connection.parent, dfir.UnusedEndMarkerComponent):
                continue
            out_type = self.type_map[out_port.data_type]

            ga_op = UnaryOp.GET_ATTR
            sub_name = in_type.get_nth_subname(i)
            # Source is a member of the input struct
            in_member_expr = HLSExpr(
                HLSExprT.UOP,
                (ga_op, sub_name),
                [
                    HLSExpr(
                        HLSExprT.VAR,
                        HLSVar(f"in_batch_i_0.data[{iterator}]", in_type),
                    )
                ],
            )

            target_var = HLSVar(f"out_batch_{out_port.name}.data[{iterator}]", out_type)
            assignments.append(CodeAssign(target_var, in_member_expr))

        return assignments

    def _translate_conditional_op(self, comp: dfir.ConditionalComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a ConditionalComponent."""
        data_port = comp.get_port("i_data")
        cond_port = comp.get_port("i_cond")
        out_port = comp.get_port("o_0")

        data_type = self.type_map[data_port.data_type]
        cond_type = self.type_map[cond_port.data_type]
        out_type = self.type_map[out_port.data_type]  # This is an Optional/Struct type

        # Source expressions
        data_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_data.data[{iterator}]", data_type))
        data_expr = HLSExpr.check_const(data_expr, data_port)
        cond_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_cond.data[{iterator}]", cond_type))
        cond_expr = HLSExpr.check_const(cond_expr, cond_port)

        # Target members of the output Optional struct
        target_data_member = HLSVar(f"out_batch_o_0.data[{iterator}].data", data_type)
        target_valid_member = HLSVar(f"out_batch_o_0.data[{iterator}].valid", cond_type)

        return [
            CodeAssign(target_data_member, data_expr),
            CodeAssign(target_valid_member, cond_expr),
        ]

    def _translate_collect_op(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """Generates a custom function body for CollectComponent due to its filtering nature."""
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp
        in_port = comp.get_port("i_0")
        out_port = comp.get_port("o_0")

        # 1. Declare local batch variables
        in_batch_var = HLSVar("in_batch_i_0", self.batch_type_map[self.type_map[in_port.data_type]])
        out_batch_var = HLSVar("out_batch_o_0", self.batch_type_map[self.type_map[out_port.data_type]])
        body.extend(
            [
                CodeVarDecl(in_batch_var.name, in_batch_var.type),
                CodeVarDecl(out_batch_var.name, out_batch_var.type),
            ]
        )

        # 2. Main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 3. Read input batch and declare output index
        in_stream_param = hls_func.params[0]  # Assume i_0 is the first param
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, in_stream_param)])
        while_loop_body.append(CodeAssign(in_batch_var, read_expr))

        out_idx_type = HLSType(HLSBasicType.UINT8)
        out_idx_var_decl = CodeVarDecl("out_idx", out_idx_type, init_val=0)  # Initialize to 0
        while_loop_body.append(out_idx_var_decl)
        out_idx_var = out_idx_var_decl.var
        # The initialization is now part of the declaration

        # 4. Inner for loop for filtering
        in_elem_type = self.type_map[in_port.data_type]  # This is an Optional type
        out_elem_type = self.type_map[out_port.data_type]

        ga_op = UnaryOp.GET_ATTR
        # Part 1: in_batch_i_0.data[i].valid
        valid_check_expr = HLSExpr(
            HLSExprT.UOP,
            (ga_op, "valid"),
            [HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type))],
        )

        # Part 2: i < in_batch_i_0.end_pos
        end_pos_check_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.LT,
            [
                HLSExpr(HLSExprT.VAR, HLSVar("i", HLSType(HLSBasicType.UINT))),
                HLSExpr(HLSExprT.UOP, (ga_op, "end_pos"), [HLSExpr(HLSExprT.VAR, in_batch_var)]),
            ],
        )

        # Combined Condition: data.valid && i < end_pos
        cond_expr = HLSExpr(HLSExprT.BINOP, BinOp.AND, [valid_check_expr, end_pos_check_expr])

        # Assignment if valid: out_batch_o_0.data[out_idx++] = in_batch_i_0.data[i].data
        assign_data = CodeAssign(
            HLSVar(f"out_batch_o_0.data[{out_idx_var.name}]", out_elem_type),
            HLSExpr(
                HLSExprT.UOP,
                (ga_op, "data"),
                [HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type))],
            ),
        )
        increment_idx = CodeAssign(
            out_idx_var,
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, out_idx_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )

        if_block = CodeIf(cond_expr, [assign_data, increment_idx])
        for_loop = CodeFor(codes=[CodePragma("UNROLL"), if_block], iter_limit="PE_NUM", iter_name="i")
        while_loop_body.append(for_loop)

        # 5. Set output batch metadata and write to stream
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_pos", out_idx_type),
                HLSExpr(HLSExprT.VAR, out_idx_var),
            )
        )
        ga_op = UnaryOp.GET_ATTR
        end_flag_expr = HLSExpr(HLSExprT.UOP, (ga_op, "end_flag"), [HLSExpr(HLSExprT.VAR, in_batch_var)])
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )

        out_stream_param = hls_func.params[1]  # Assume o_0 is the second param
        while_loop_body.append(CodeWriteStream(out_stream_param, out_batch_var))

        # 6. Break condition
        while_loop_body.append(CodeIf(end_flag_expr, [CodeBreak()]))

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))
        return body

    # --- Reduce Component Translation Logic ---

    # In graphyflow/backend_manager.py, replace the existing _inline_sub_graph_logic function

    def _inline_sub_graph_logic(
        self,
        start_ports: List[dfir.Port],
        end_port: dfir.Port,
        io_var_map: Dict[dfir.Port, HLSVar],
    ) -> List[HLSCodeLine]:
        """
        Traverses a sub-graph from start to end ports and generates the inlined logic.
        This version uses the robust _add_pod_to_float_cast helper.
        """
        code_lines: List[HLSCodeLine] = []
        p2var_map = io_var_map.copy()
        code_lines.append(CodeComment(" -- Inline sub graph --"))

        q = [p.connection.parent for p in start_ports if p.connected]
        visited_ids = set([c.readable_id for c in q])
        head = 0
        while head < len(q):
            comp = q[head]
            head += 1
            code_lines.append(CodeComment(f"Starting for comp {comp.name}"))
            inputs_ready = all(p.connection in p2var_map for p in comp.in_ports)
            if not inputs_ready:
                q.append(comp)
                if head > len(q) * 2 + len(start_ports) * 2:
                    raise RuntimeError(f"Deadlock in sub-graph topological sort at component {comp.name}")
                continue

            for out_port in comp.out_ports:
                if out_port.connected:
                    if out_port.connection == end_port:
                        p2var_map[out_port] = p2var_map[end_port]
                    else:
                        temp_var = HLSVar(
                            f"temp_{out_port.parent.name}_{out_port.name}", self.type_map[out_port.data_type]
                        )
                        code_lines.append(CodeVarDecl(temp_var.name, temp_var.type))
                        p2var_map[out_port] = temp_var

            if isinstance(comp, dfir.BinOpComponent):
                op1_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection])
                op1_expr = HLSExpr.check_const(op1_expr, comp.get_port("i_0"))
                op2_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_1").connection])
                op2_expr = HLSExpr.check_const(op2_expr, comp.get_port("i_1"))
                target_var = p2var_map[comp.get_port("o_0")]

                if op1_expr.val.type.type == HLSBasicType.AP_FIXED_POD:
                    is_comparison = comp.op in [BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.GT, BinOp.LE, BinOp.GE]

                    final_op1_expr = self._add_pod_to_float_cast(
                        op1_expr, code_lines, f"lhs_{comp.readable_id}"
                    )
                    final_op2_expr = self._add_pod_to_float_cast(
                        op2_expr, code_lines, f"rhs_{comp.readable_id}"
                    )

                    op_expr = HLSExpr(HLSExprT.BINOP, comp.op, [final_op1_expr, final_op2_expr])

                    if is_comparison:
                        code_lines.append(CodeAssign(target_var, op_expr))
                    else:
                        ap_fixed_type = HLSType(HLSBasicType.FLOAT)
                        # Your fix for variable naming is integrated here
                        result_var = HLSVar(f"temp_{comp.name}_o_0_ap_result", ap_fixed_type)
                        code_lines.append(CodeVarDecl(result_var.name, result_var.type))
                        code_lines.append(CodeAssign(result_var, op_expr))
                        self._add_float_to_pod_cast(result_var, code_lines, target_var)
                else:
                    expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1_expr, op2_expr])
                    code_lines.append(CodeAssign(target_var, expr))

            elif isinstance(comp, dfir.UnaryOpComponent):
                op1 = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection])
                op1 = HLSExpr.check_const(op1, comp.get_port("i_0"))
                comp_op_var = comp.op
                if comp.op in [UnaryOp.GET_ATTR, UnaryOp.SELECT]:
                    assert op1.val.type.type == HLSBasicType.STRUCT
                    comp_op_var = (comp_op_var, comp.select_index)
                expr = HLSExpr(HLSExprT.UOP, comp_op_var, [op1])
                code_lines.append(CodeAssign(p2var_map[comp.get_port("o_0")], expr))
            elif isinstance(comp, dfir.CopyComponent):
                in_var_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection])
                in_var_expr = HLSExpr.check_const(in_var_expr, comp.get_port("i_0"))
                target_o0 = p2var_map[comp.get_port("o_0")]
                target_o1 = p2var_map[comp.get_port("o_1")]
                code_lines.append(CodeAssign(target_o0, in_var_expr))
                code_lines.append(CodeAssign(target_o1, in_var_expr))
            elif isinstance(comp, dfir.GatherComponent):
                target_struct_var = p2var_map[comp.get_port("o_0")]
                for i, in_port in enumerate(comp.in_ports):
                    in_var_expr = HLSExpr(HLSExprT.VAR, p2var_map[in_port.connection])
                    in_var_expr = HLSExpr.check_const(in_var_expr, in_port)
                    member_var = HLSVar(f"{target_struct_var.name}.ele_{i}", in_var_expr.val.type)
                    code_lines.append(CodeAssign(member_var, in_var_expr))
            elif isinstance(comp, dfir.ScatterComponent):
                in_var = p2var_map[comp.get_port("i_0").connection]
                for i, out_port in enumerate(comp.out_ports):
                    if isinstance(out_port.connection.parent, dfir.UnusedEndMarkerComponent):
                        continue
                    ga_op = UnaryOp.GET_ATTR
                    sub_name = in_var.type.get_nth_subname(i)
                    expr = HLSExpr(HLSExprT.UOP, (ga_op, sub_name), [HLSExpr(HLSExprT.VAR, in_var)])
                    code_lines.append(CodeAssign(p2var_map[out_port], expr))

            elif isinstance(comp, dfir.ConditionalComponent):
                data_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_data").connection])
                data_expr = HLSExpr.check_const(data_expr, comp.get_port("i_data"))
                cond_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_cond").connection])
                cond_expr = HLSExpr.check_const(cond_expr, comp.get_port("i_cond"))
                target_struct_var = p2var_map[comp.get_port("o_0")]
                assign_data = CodeAssign(
                    HLSVar(f"{target_struct_var.name}.data", data_expr.val.type), data_expr
                )
                assign_valid = CodeAssign(
                    HLSVar(f"{target_struct_var.name}.valid", cond_expr.val.type), cond_expr
                )
                code_lines.extend([assign_data, assign_valid])

            elif isinstance(comp, dfir.CollectComponent):
                in_opt_var = p2var_map[comp.get_port("i_0").connection]
                out_var = p2var_map[comp.get_port("o_0")]
                valid_op = UnaryOp.GET_ATTR
                cond_expr = HLSExpr(HLSExprT.UOP, (valid_op, "valid"), [HLSExpr(HLSExprT.VAR, in_opt_var)])
                data_op = UnaryOp.GET_ATTR
                assign_expr = HLSExpr(HLSExprT.UOP, (data_op, "data"), [HLSExpr(HLSExprT.VAR, in_opt_var)])
                if_block = CodeIf(cond_expr, [CodeAssign(out_var, assign_expr)])
                code_lines.append(if_block)
            elif isinstance(comp, dfir.UnusedEndMarkerComponent):
                pass
            else:
                code_lines.append(CodeComment(f"Inlined logic for {comp.__class__} ({comp.name})"))

            for p in comp.out_ports:
                if p.connected and not isinstance(
                    p.connection.parent,
                    (dfir.ReduceComponent, dfir.UnusedEndMarkerComponent),
                ):
                    successor_comp = p.connection.parent
                    if successor_comp.readable_id not in visited_ids:
                        q.append(successor_comp)
                        visited_ids.add(successor_comp.readable_id)

        code_lines.append(CodeComment(" -- Inline sub graph end --"))
        return code_lines

    def _translate_reduce_preprocess_op(self, comp: dfir.ReduceComponent, iterator: str) -> List[HLSCodeLine]:
        """
        Generates the inner-loop logic for ReduceComponent's pre_process stage.
        """
        in_type = self.type_map[comp.get_port("i_0").data_type]
        key_out_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
        transform_out_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]

        in_elem_var = HLSVar(f"in_batch_i_0.data[{iterator}]", in_type)
        key_out_elem_var = HLSVar("key_out_elem", key_out_type)
        transform_out_elem_var = HLSVar("transform_out_elem", transform_out_type)
        code_lines = [
            CodeVarDecl(key_out_elem_var.name, key_out_elem_var.type),
            CodeVarDecl(transform_out_elem_var.name, transform_out_elem_var.type),
        ]

        key_sub_graph_start = comp.get_port("o_reduce_key_in")
        key_sub_graph_end = comp.get_port("i_reduce_key_out")
        key_io_map = {key_sub_graph_start: in_elem_var, key_sub_graph_end: key_out_elem_var}
        code_lines.extend(self._inline_sub_graph_logic([key_sub_graph_start], key_sub_graph_end, key_io_map))

        transform_sub_graph_start = comp.get_port("o_reduce_transform_in")
        transform_sub_graph_end = comp.get_port("i_reduce_transform_out")
        transform_io_map = {
            transform_sub_graph_start: in_elem_var,
            transform_sub_graph_end: transform_out_elem_var,
        }
        code_lines.extend(
            self._inline_sub_graph_logic(
                [transform_sub_graph_start], transform_sub_graph_end, transform_io_map
            )
        )

        assign_key = CodeAssign(
            HLSVar(f"out_batch_intermediate_key.data[{iterator}]", key_out_type),
            HLSExpr(HLSExprT.VAR, key_out_elem_var),
        )
        assign_transform = CodeAssign(
            HLSVar(f"out_batch_intermediate_transform.data[{iterator}]", transform_out_type),
            HLSExpr(HLSExprT.VAR, transform_out_elem_var),
        )
        code_lines.extend([assign_key, assign_transform])

        return code_lines

    def _translate_reduce_preprocess(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """
        Generates a custom body for the pre_process stage, correctly handling its unique I/O.
        """
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # 1. Get HLSVar for each parameter from the function signature
        in_stream, key_stream, transform_stream = hls_func.params

        # 2. Declare local batch variables for I/O
        in_batch_var = HLSVar("in_batch_i_0", in_stream.type.sub_types[0])
        key_out_batch_var = HLSVar("out_batch_intermediate_key", key_stream.type.sub_types[0])
        transform_out_batch_var = HLSVar(
            "out_batch_intermediate_transform", transform_stream.type.sub_types[0]
        )

        body.extend(
            [
                CodeVarDecl(in_batch_var.name, in_batch_var.type),
                CodeVarDecl(key_out_batch_var.name, key_out_batch_var.type),
                CodeVarDecl(transform_out_batch_var.name, transform_out_batch_var.type),
            ]
        )

        end_flag_var = HLSVar("end_flag", HLSType(HLSBasicType.BOOL))
        body.append(CodeVarDecl(end_flag_var.name, end_flag_var.type))

        # 3. Build the main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 4. Read input batch
        while_loop_body.append(
            CodeAssign(
                in_batch_var,
                HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, in_stream)]),
            )
        )

        # 5. Build the inner for-loop with the core logic
        inner_logic = self._translate_reduce_preprocess_op(comp, "i")
        for_loop = CodeFor(codes=[CodePragma("UNROLL")] + inner_logic, iter_limit="PE_NUM", iter_name="i")
        while_loop_body.append(for_loop)

        # 6. Copy metadata (end_flag, end_pos) from input batch to both output batches
        ga_op = UnaryOp.GET_ATTR
        end_flag_expr = HLSExpr(HLSExprT.UOP, (ga_op, "end_flag"), [HLSExpr(HLSExprT.VAR, in_batch_var)])
        end_pos_expr = HLSExpr(HLSExprT.UOP, (ga_op, "end_pos"), [HLSExpr(HLSExprT.VAR, in_batch_var)])

        # Assign metadata to key_out_batch
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{key_out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{key_out_batch_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                end_pos_expr,
            )
        )

        # Assign metadata to transform_out_batch
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{transform_out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{transform_out_batch_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                end_pos_expr,
            )
        )

        # 7. Write both output batches to their streams
        while_loop_body.append(CodeWriteStream(key_stream, key_out_batch_var))
        while_loop_body.append(CodeWriteStream(transform_stream, transform_out_batch_var))

        # 8. Check for break condition
        while_loop_body.append(CodeAssign(end_flag_var, end_flag_expr))
        while_loop_body.append(CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()]))

        # 9. Finalize the function body
        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))
        return body

    def _translate_reduce_unit_reduce(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """
        Generates the body for the second stage of Reduce (stateful accumulation).
        This version is updated to use a more performant circular buffer for draining.
        """
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # 1. Get types and variables from the new function signature
        kt_streams, out_streams = hls_func.params
        kt_type = kt_streams.type.sub_types[0].sub_types[0]
        key_type = kt_streams.type.sub_types[0].sub_types[0].sub_types[0].sub_types[0]
        transform_type = kt_streams.type.sub_types[0].sub_types[0].sub_types[0].sub_types[1]
        single_out_stream_type = out_streams.type
        # The output from this unit is a batched stream
        out_batch_type = single_out_stream_type.sub_types[0]
        out_data_type = out_batch_type.sub_types[0].sub_types[0]

        bram_elem_type = self._to_hls_type(
            dftype.TupleType([comp.get_port("i_reduce_transform_out").data_type, dftype.BoolType()])
        )

        body.append(CodeComment("1. Stateful memories for PE_NUM parallel reduction units"))
        key_mem_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[bram_elem_type], array_dims=["PE_NUM", "MAX_NUM"]
        )
        body.append(CodeVarDecl("key_mem", key_mem_type))
        body.append(CodePragma("BIND_STORAGE variable=key_mem type=RAM_2P impl=URAM"))
        body.append(CodePragma("ARRAY_PARTITION variable=key_mem complete dim=1"))

        key_buffer_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[bram_elem_type], array_dims=["PE_NUM", "L + 1"]
        )
        body.append(CodeVarDecl("key_buffer", key_buffer_type))
        body.append(CodePragma("ARRAY_PARTITION variable=key_buffer complete dim=0"))

        i_buffer_base_type = HLSType(HLSBasicType.UINT)
        i_buffer_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[i_buffer_base_type], array_dims=["PE_NUM", "L + 1"]
        )
        body.append(CodeVarDecl("i_buffer", i_buffer_type))
        body.append(CodePragma("ARRAY_PARTITION variable=i_buffer complete dim=0"))

        body.append(CodeComment("2. Memory initialization for all PEs"))
        uint_type = HLSType(HLSBasicType.UINT)
        max_num_var = HLSVar("MAX_NUM", uint_type)
        assign_val_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.ADD,
            [HLSExpr(HLSExprT.VAR, max_num_var), HLSExpr(HLSExprT.CONST, 1)],
        )
        assign_ibuf = CodeAssign(HLSVar("i_buffer[pe][i]", uint_type), assign_val_expr)
        clear_ibuf_inner_loop = CodeFor([CodePragma("UNROLL"), assign_ibuf], "L + 1", iter_name="i")
        clear_ibuf_outer_loop = CodeFor(
            [CodePragma("UNROLL"), clear_ibuf_inner_loop], "PE_NUM", iter_name="pe"
        )
        body.append(clear_ibuf_outer_loop)

        target_valid_flag = HLSVar("key_mem[pe][i].ele_1", HLSType(HLSBasicType.BOOL))
        assign_valid_false = CodeAssign(target_valid_flag, HLSExpr(HLSExprT.CONST, False))
        clear_valid_inner_loop = CodeFor([CodePragma("UNROLL"), assign_valid_false], "MAX_NUM", iter_name="i")
        clear_valid_outer_loop = CodeFor(
            [CodePragma("UNROLL"), clear_valid_inner_loop], "PE_NUM", iter_name="pe"
        )
        body.append(clear_valid_outer_loop)

        body.append(CodeComment("3. Main processing loop for aggregation across PEs"))
        end_flag_var = HLSVar("end_flag", HLSType(HLSBasicType.BOOL))
        body.append(CodeVarDecl(end_flag_var.name, end_flag_var.type))
        all_end_flag_var = HLSVar(
            "all_end_flags",
            HLSType(HLSBasicType.ARRAY, [end_flag_var.type], array_dims=["PE_NUM"]),
        )
        body.append(CodeVarDecl(all_end_flag_var.name, all_end_flag_var.type))
        body.append(CodePragma("ARRAY_PARTITION variable=all_end_flags complete dim=0"))
        assign_end_flag = CodeAssign(
            HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type),
            HLSExpr(HLSExprT.CONST, False),
        )
        reset_end_loop = CodeFor(
            codes=[CodePragma("UNROLL"), assign_end_flag],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        body.append(reset_end_loop)

        while_loop_body = [CodePragma("PIPELINE")]
        kt_elem_var = HLSVar("kt_elem", kt_type)
        key_elem_var = HLSVar("key_elem", key_type)
        transform_elem_var = HLSVar("transform_elem", transform_type)
        while_loop_body.extend(
            [
                CodeVarDecl(kt_elem_var.name, kt_elem_var.type),
                CodeVarDecl(key_elem_var.name, key_elem_var.type),
                CodeVarDecl(transform_elem_var.name, transform_elem_var.type),
            ]
        )
        inner_loop_logic = self._translate_reduce_unit_inner_loop(
            comp, bram_elem_type, "i", key_elem_var, transform_elem_var
        )

        kt_stream_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"{kt_streams.name}[i]", kt_type))
        read_kt = CodeAssign(kt_elem_var, HLSExpr(HLSExprT.STREAM_READ, None, [kt_stream_expr]))
        key_from_wrapper = HLSExpr(
            HLSExprT.UOP,
            (UnaryOp.GET_ATTR, "data.key"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        transform_from_wrapper = HLSExpr(
            HLSExprT.UOP,
            (UnaryOp.GET_ATTR, "data.transform"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        read_key = CodeAssign(key_elem_var, key_from_wrapper)
        read_transform = CodeAssign(transform_elem_var, transform_from_wrapper)
        end_flag_from_wrapper = HLSExpr(
            HLSExprT.UOP,
            (UnaryOp.GET_ATTR, "end_flag"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        end_flag_in_array = HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type)

        process_logic = CodeIf(
            expr=end_flag_from_wrapper,
            if_codes=[CodeAssign(end_flag_in_array, end_flag_from_wrapper)],
            else_codes=[read_key, read_transform] + inner_loop_logic,
        )
        read_and_process_block = CodeIf(
            expr=HLSExpr(
                HLSExprT.BINOP,
                BinOp.AND,
                [
                    HLSExpr(HLSExprT.UOP, UnaryOp.NOT, [HLSExpr(HLSExprT.VAR, end_flag_in_array)]),
                    HLSExpr(
                        HLSExprT.UOP, UnaryOp.NOT, [HLSExpr(HLSExprT.STREAM_EMPTY, None, [kt_stream_expr])]
                    ),
                ],
            ),
            if_codes=[read_kt, process_logic],
        )
        pe_processing_loop = CodeFor(
            codes=[CodePragma("UNROLL"), read_and_process_block],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(pe_processing_loop)

        # Check for end condition (all PEs must have seen an end flag)
        check_end_flag_logic = [CodeAssign(end_flag_var, HLSExpr(HLSExprT.CONST, True))]
        and_comp_vars = [
            HLSExpr(HLSExprT.VAR, end_flag_var),
            HLSExpr(HLSExprT.VAR, end_flag_in_array),
        ]
        end_flag_agg_loop = CodeFor(
            codes=[
                CodePragma("UNROLL"),
                CodeAssign(
                    end_flag_var,
                    HLSExpr(HLSExprT.BINOP, BinOp.AND, and_comp_vars),
                ),
            ],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        check_end_flag_logic.append(end_flag_agg_loop)
        check_end_flag_logic.append(CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()]))
        while_loop_body.extend(check_end_flag_logic)

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))

        body.append(CodeComment("4. Final output loop to drain all PE memories with swapped loops"))

        # New variable declarations for circular buffer draining
        cnt_var = HLSVar("data_cnt", HLSType(HLSBasicType.UINT))
        body.append(CodeVarDecl(cnt_var.name, cnt_var.type, init_val=0))
        body.append(CodeAssign(cnt_var, HLSExpr(HLSExprT.CONST, 0)))

        start_pos_var = HLSVar("start_pos", HLSType(HLSBasicType.UINT))
        body.append(CodeVarDecl(start_pos_var.name, start_pos_var.type, init_val=0))
        body.append(CodeAssign(start_pos_var, HLSExpr(HLSExprT.CONST, 0)))

        data_pack_var = HLSVar("data_pack", out_batch_type)
        body.append(CodeVarDecl(data_pack_var.name, data_pack_var.type))
        body.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                HLSExpr(HLSExprT.CONST, False),
            )
        )

        drain_buffer_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[out_data_type], array_dims=[f"((PE_NUM << 1))"]
        )
        drain_buffer_var = HLSVar("data_to_write", drain_buffer_type)
        body.append(CodeVarDecl(drain_buffer_var.name, drain_buffer_var.type))
        body.append(CodePragma(f"ARRAY_PARTITION variable={drain_buffer_var.name} complete dim=0"))

        # Since CodeFor doesn't support custom increments, we simulate it with a while loop
        k_var = HLSVar("k", HLSType(HLSBasicType.UINT))
        body.append(CodeVarDecl(k_var.name, k_var.type, init_val=0))
        body.append(CodeAssign(k_var, HLSExpr(HLSExprT.CONST, 0)))

        # Inner logic of the tiled draining loop
        pe_var = HLSVar("pe", HLSType(HLSBasicType.UINT))
        k_plus_pe_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.ADD,
            [HLSExpr(HLSExprT.VAR, k_var), HLSExpr(HLSExprT.VAR, pe_var)],
        )

        is_valid_expr = HLSExpr(
            HLSExprT.VAR,
            HLSVar(f"key_mem[pe][{k_plus_pe_expr.code}].ele_1", HLSType(HLSBasicType.BOOL)),
        )
        data_from_mem = HLSExpr(
            HLSExprT.VAR, HLSVar(f"key_mem[pe][{k_plus_pe_expr.code}].ele_0", out_data_type)
        )

        write_to_buffer_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.MOD,
            [HLSExpr(HLSExprT.VAR, start_pos_var), HLSExpr(HLSExprT.CONST, "((PE_NUM << 1))")],
        )
        assign_to_buffer = CodeAssign(
            HLSVar(f"{drain_buffer_var.name}[{write_to_buffer_expr.code}]", out_data_type),
            data_from_mem,
        )

        increment_cnt = CodeAssign(
            cnt_var,
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )
        increment_pos = CodeAssign(
            start_pos_var,
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, start_pos_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )

        if_valid_block = CodeIf(is_valid_expr, [assign_to_buffer, increment_cnt, increment_pos])

        drain_inner_pe_loop = CodeFor(
            codes=[CodePragma("UNROLL"), if_valid_block], iter_limit="PE_NUM", iter_name="pe"
        )

        # Logic to pack and write a full batch
        pack_and_write_logic = []
        pack_and_write_logic.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                HLSExpr(HLSExprT.CONST, "PE_NUM"),
            )
        )

        i_var = HLSVar("i", HLSType(HLSBasicType.UINT))

        read_from_buffer_offset = HLSExpr(
            HLSExprT.BINOP,
            BinOp.ADD,
            [
                HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.SUB,
                    [HLSExpr(HLSExprT.VAR, start_pos_var), HLSExpr(HLSExprT.VAR, cnt_var)],
                ),
                HLSExpr(HLSExprT.VAR, i_var),
            ],
        )
        read_from_buffer_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.MOD,
            [read_from_buffer_offset, HLSExpr(HLSExprT.CONST, "(PE_NUM << 1)")],
        )

        packing_loop_body = [
            CodePragma("UNROLL"),
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.data[i]", out_data_type),
                HLSExpr(
                    HLSExprT.VAR,
                    HLSVar(f"{drain_buffer_var.name}[{read_from_buffer_expr.code}]", out_data_type),
                ),
            ),
        ]
        packing_loop = CodeFor(packing_loop_body, iter_limit="PE_NUM", iter_name="i")
        pack_and_write_logic.append(packing_loop)
        pack_and_write_logic.append(CodeWriteStream(out_streams, data_pack_var))
        pack_and_write_logic.append(
            CodeAssign(
                cnt_var,
                HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.SUB,
                    [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, "PE_NUM")],
                ),
            )
        )

        if_data_full = CodeIf(
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.GE,
                [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, "PE_NUM")],
            ),
            pack_and_write_logic,
        )

        # Main draining while loop
        drain_while_body = [
            CodePragma("PIPELINE"),
            drain_inner_pe_loop,
            if_data_full,
            CodeAssign(
                k_var,
                HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.ADD,
                    [HLSExpr(HLSExprT.VAR, k_var), HLSExpr(HLSExprT.CONST, "PE_NUM")],
                ),
            ),
        ]
        body.append(
            CodeWhile(
                codes=drain_while_body,
                iter_expr=HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.LT,
                    [HLSExpr(HLSExprT.VAR, k_var), HLSExpr(HLSExprT.CONST, "MAX_NUM")],
                ),
            )
        )

        body.append(CodeComment("5. Drain any remaining data and send final batch with end_flag"))

        # Final packing logic
        final_pack_logic = []
        final_pack_logic.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                HLSExpr(HLSExprT.CONST, True),
            )
        )
        final_pack_logic.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                HLSExpr(HLSExprT.VAR, cnt_var),
            )
        )
        d_buffer_sub = HLSVar(f"{drain_buffer_var.name}[{read_from_buffer_expr.code}]", out_data_type)
        data_assign = CodeAssign(
            HLSVar(f"{data_pack_var.name}.data[i]", out_data_type),
            HLSExpr(HLSExprT.VAR, d_buffer_sub),
        )
        lt_cmp_vars = [HLSExpr(HLSExprT.VAR, i_var), HLSExpr(HLSExprT.VAR, cnt_var)]
        final_packing_loop_body = [
            CodePragma("UNROLL"),
            CodeIf(
                HLSExpr(HLSExprT.BINOP, BinOp.LT, lt_cmp_vars),
                [data_assign],
            ),
        ]
        final_packing_loop = CodeFor(final_packing_loop_body, iter_limit="PE_NUM", iter_name="i")

        final_pack_logic.append(final_packing_loop)
        final_pack_logic.append(CodeWriteStream(out_streams, data_pack_var))

        body.extend(final_pack_logic)

        return body

    def _translate_reduce_unit_inner_loop(
        self,
        comp: dfir.ReduceComponent,
        bram_elem_type: HLSType,
        pe_idx: str,
        key_var,
        val_var,
    ) -> List[HLSCodeLine]:
        """
        Helper to generate the complex logic inside unit_reduce's PE_NUM loop.
        *** MODIFIED to accept a PE index ***
        """
        key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
        accum_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]
        bool_type = HLSType(HLSBasicType.BOOL)

        # # 1. Get current key and value from the batch using the PE index
        # key_var = HLSVar("current_key", key_type)
        # val_var = HLSVar("current_val", accum_type)
        # logic = [
        #     CodeVarDecl(key_var.name, key_var.type),
        #     CodeVarDecl(val_var.name, val_var.type),
        #     CodeAssign(
        #         key_var, HLSExpr(HLSExprT.VAR, HLSVar(f"in_key_batch.data[{pe_idx}]", key_type))
        #     ),
        #     CodeAssign(
        #         val_var,
        #         HLSExpr(HLSExprT.VAR, HLSVar(f"in_transform_batch.data[{pe_idx}]", accum_type)),
        #     ),
        # ]
        logic = []

        # *** 关键修改: 所有对内存的访问都使用 pe_idx 作为第一维度 ***
        # 2. Read old element from this PE's BRAM & buffer
        old_ele_var = HLSVar("old_ele", bram_elem_type)
        logic.append(CodeVarDecl(old_ele_var.name, old_ele_var.type))
        logic.append(
            CodeAssign(
                old_ele_var,
                HLSExpr(HLSExprT.VAR, HLSVar(f"key_mem[{pe_idx}][{key_var.name}]", bram_elem_type)),
            )
        )

        # 3. Buffer management for this PE
        buffer_elem_expr = HLSExpr(
            HLSExprT.VAR, HLSVar(f"i_buffer[{pe_idx}][i_search]", HLSType(HLSBasicType.UINT))
        )
        if_condition = HLSExpr(HLSExprT.BINOP, BinOp.EQ, [HLSExpr(HLSExprT.VAR, key_var), buffer_elem_expr])
        value_to_assign = HLSExpr(HLSExprT.VAR, HLSVar(f"key_buffer[{pe_idx}][i_search]", bram_elem_type))
        search_loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeIf(if_condition, [CodeAssign(old_ele_var, value_to_assign)]),
            ],
            "L + 1",
            iter_name="i_search",
        )
        logic.append(search_loop)

        i_buffer_dest = HLSVar(f"i_buffer[{pe_idx}][i_move]", HLSType(HLSBasicType.UINT))
        i_buffer_src = HLSExpr(
            HLSExprT.VAR, HLSVar(f"i_buffer[{pe_idx}][i_move + 1]", HLSType(HLSBasicType.UINT))
        )
        key_buffer_dest = HLSVar(f"key_buffer[{pe_idx}][i_move]", bram_elem_type)
        key_buffer_src = HLSExpr(HLSExprT.VAR, HLSVar(f"key_buffer[{pe_idx}][i_move + 1]", bram_elem_type))
        shift_loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeBlock(
                    [
                        CodeAssign(i_buffer_dest, i_buffer_src),
                        CodeAssign(key_buffer_dest, key_buffer_src),
                    ]
                ),
            ],
            "L",
            iter_name="i_move",
        )
        logic.append(shift_loop)

        # 4. If/Else logic for aggregation (logic itself is unchanged)
        new_ele_var = HLSVar("new_ele", bram_elem_type)
        logic.append(CodeVarDecl(new_ele_var.name, new_ele_var.type))
        is_valid_expr = HLSExpr(
            HLSExprT.UOP, (UnaryOp.GET_ATTR, "ele_1"), [HLSExpr(HLSExprT.VAR, old_ele_var)]
        )
        if_codes = [
            CodeAssign(HLSVar(f"{new_ele_var.name}.ele_1", bool_type), HLSExpr(HLSExprT.CONST, True)),
            CodeAssign(HLSVar(f"{new_ele_var.name}.ele_0", accum_type), HLSExpr(HLSExprT.VAR, val_var)),
        ]
        old_data_var = HLSVar("old_data", accum_type)
        unit_res_var = HLSVar(f"{new_ele_var.name}.ele_0", accum_type)
        unit_starts = [
            comp.get_port("o_reduce_unit_start_0"),
            comp.get_port("o_reduce_unit_start_1"),
        ]
        unit_end = comp.get_port("i_reduce_unit_end")
        io_map = {unit_starts[0]: old_data_var, unit_starts[1]: val_var, unit_end: unit_res_var}
        else_codes = [
            CodeVarDecl(old_data_var.name, old_data_var.type),
            CodeAssign(
                old_data_var,
                HLSExpr(
                    HLSExprT.UOP,
                    (UnaryOp.GET_ATTR, "ele_0"),
                    [HLSExpr(HLSExprT.VAR, old_ele_var)],
                ),
            ),
            *self._inline_sub_graph_logic(unit_starts, unit_end, io_map),
            CodeAssign(HLSVar(f"{new_ele_var.name}.ele_1", bool_type), HLSExpr(HLSExprT.CONST, True)),
        ]
        logic.append(CodeIf(is_valid_expr, if_codes=else_codes, else_codes=if_codes))

        # 5. Write back to this PE's BRAM and buffer
        logic.append(
            CodeAssign(
                HLSVar(f"key_mem[{pe_idx}][{key_var.name}]", bram_elem_type),
                HLSExpr(HLSExprT.VAR, new_ele_var),
            )
        )
        logic.append(
            CodeAssign(
                HLSVar(f"key_buffer[{pe_idx}][L]", bram_elem_type),
                HLSExpr(HLSExprT.VAR, new_ele_var),
            )
        )
        logic.append(
            CodeAssign(
                HLSVar(f"i_buffer[{pe_idx}][L]", HLSType(HLSBasicType.UINT)),
                HLSExpr(HLSExprT.VAR, key_var),
            )
        )

        return logic

    # ======================================================================== #
    #                            PHASE 4: Final Assembly                       #
    # ======================================================================== #

    def _get_host_output_type(self) -> HLSType:
        """
        A helper function to define and return the host-friendly output batch type.
        This centralizes the definition of KernelOutputBatch and registers the types.
        """
        kernel_output_data_type = HLSType(
            HLSBasicType.STRUCT,
            sub_types=[HLSType(HLSBasicType.REAL_FLOAT), HLSType(HLSBasicType.INT)],
            struct_prop_names=["distance", "id"],
            struct_name="KernelOutputData",
        )
        if kernel_output_data_type.name not in self.struct_definitions:
            self.struct_definitions[kernel_output_data_type.name] = (
                kernel_output_data_type,
                ["distance", "id"],
            )

        output_data_array_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[kernel_output_data_type], array_dims=["PE_NUM"]
        )
        host_output_type = HLSType(
            HLSBasicType.STRUCT,
            sub_types=[output_data_array_type, HLSType(HLSBasicType.BOOL), HLSType(HLSBasicType.UINT8)],
            struct_prop_names=["data", "end_flag", "end_pos"],
            struct_name="KernelOutputBatch",
        )
        if host_output_type.name not in self.struct_definitions:
            self.struct_definitions[host_output_type.name] = (
                host_output_type,
                ["data", "end_flag", "end_pos"],
            )

        return host_output_type

    def _topologically_sort_structs(self) -> List[Tuple[HLSType, List[str]]]:
        """Sorts struct definitions based on their member dependencies."""
        from collections import defaultdict

        adj = defaultdict(list)
        in_degree = defaultdict(int)

        all_struct_names = self.struct_definitions.keys()

        # Build dependency graph
        for dependent_struct_name, (hls_type, _) in self.struct_definitions.items():
            if dependent_struct_name not in in_degree:
                in_degree[dependent_struct_name] = 0

            if not hls_type.sub_types:
                continue

            for member_type in hls_type.sub_types:

                base_member_type = member_type
                if base_member_type.type == HLSBasicType.ARRAY:
                    base_member_type = base_member_type.sub_types[0]

                if base_member_type.type == HLSBasicType.STRUCT:
                    dependency_struct_name = base_member_type.name

                    if dependency_struct_name in all_struct_names:
                        # --- *** 关键修正：修复依赖关系 *** ---
                        # The dependent_struct depends on the dependency_struct.
                        # The edge is: dependency_struct -> dependent_struct.
                        adj[dependency_struct_name].append(dependent_struct_name)
                        in_degree[dependent_struct_name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name in self.struct_definitions if in_degree[name] == 0]
        sorted_structs = []

        while queue:
            u = queue.pop(0)
            if u in self.struct_definitions:
                sorted_structs.append(self.struct_definitions[u])
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(sorted_structs) != len(self.struct_definitions):
            print("--- DEBUG: Cycle detected in struct dependencies ---")
            print("Total structs:", len(self.struct_definitions))
            print("Sorted structs:", len(sorted_structs))
            print("Remaining in_degrees:", {k: v for k, v in in_degree.items() if v > 0})
            raise RuntimeError("A cycle was detected in the struct definitions.")

        return sorted_structs

    def _generate_header_file(self, top_func_name: str, top_func_sig: str) -> str:
        """Generates the full content of the .h header file."""
        header_guard = f"__GRAPHYFLOW_{top_func_name.upper()}_H__"
        code = f"#ifndef {header_guard}\n#define {header_guard}\n\n"
        code += "#include <hls_stream.h>\n#include <ap_fixed.h>\n#include <stdint.h>\n\n"
        code += f"#define PE_NUM {self.PE_NUM}\n"
        code += f"#define MAX_NUM {self.MAX_NUM}\n"
        code += f"#define L {self.L}\n\n"

        code += "// --- Struct Type Definitions ---\n"
        sorted_defs = self._topologically_sort_structs()
        for hls_type, members in sorted_defs:
            code += hls_type.gen_decl(members) + "\n"

        code += "// --- Function Prototypes ---\n"
        for func in self.hls_functions.values():
            params_str = ", ".join([p.type.get_upper_param(p.name, True) for p in func.params])
            code += f"void {func.name}({params_str});\n"

        code += f"\n// --- Top-Level Function Prototype ---\n"
        code += f"{top_func_sig};\n\n"

        code += f"#endif // {header_guard}\n"
        return code

    def _generate_top_level_function_body(self) -> List[HLSCodeLine]:
        """Generates the implementation of the dataflow core function body,
        based on the proven logic from the original implementation."""
        from collections import defaultdict

        body: List[HLSCodeLine] = [CodePragma("DATAFLOW")]

        # 1. Declare all intermediate streams
        for decl, pragma in self.top_level_stream_decls:
            body.append(decl)
            if pragma:
                body.append(pragma)

        # 2. Prepare maps for I/O and intermediate streams
        stream_map = {decl.var.name: decl.var for decl, _ in self.top_level_stream_decls}
        top_io_map: Dict[int, HLSVar] = {}

        # --- *** 关键修正：在此处重新发现顶层IO端口，确保状态最新 *** ---
        # This logic is taken directly from the user-provided working code.
        top_level_inputs = []
        for comp in self.comp_col_store.components:
            if isinstance(comp, dfir.IOComponent) and comp.io_type == dfir.IOComponent.IOType.INPUT:
                if comp.get_port("o_0").connected:
                    top_level_inputs.append(comp.get_port("o_0").connection)
        top_level_outputs = self.comp_col_store.outputs

        current_top_level_io_ports = top_level_inputs + top_level_outputs
        # ----------------------------------------------------------------

        # Populate top_io_map using the fresh list of ports
        for p in current_top_level_io_ports:
            # Note: For ArrayType, we need its inner type for batching.
            dfir_type = p.data_type.type_ if isinstance(p.data_type, dftype.ArrayType) else p.data_type
            batch_type = self.batch_type_map[self.type_map[dfir_type]]
            top_io_map[p.readable_id] = HLSVar(
                f"{p.unique_name}_stream", HLSType(HLSBasicType.STREAM, [batch_type])
            )

        # 3. Topologically sort functions (using the user-provided, proven logic)
        stream_funcs = [f for f in self.hls_functions.values() if f.streamed]
        id_to_func = {f.readable_id: f for f in stream_funcs}
        comp_to_func = {f.dfir_comp: f for f in stream_funcs}  # This is safe within this function's context
        reduce_comp_to_pre = {}

        adj = defaultdict(list)
        in_degree = {f.readable_id: 0 for f in stream_funcs}

        for func in stream_funcs:
            if isinstance(func.dfir_comp, dfir.ReduceComponent) and "pre_process" in func.name:
                in_port = func.dfir_comp.get_port("i_0")
                if in_port.connected and in_port.connection.parent in comp_to_func:
                    adj[comp_to_func[in_port.connection.parent].readable_id].append(func.readable_id)
                    in_degree[func.readable_id] += 1
                reduce_comp_to_pre[func.dfir_comp] = func

        for func in stream_funcs:
            comp = func.dfir_comp
            if isinstance(comp, dfir.ReduceComponent):
                if "pre_process" in func.name:
                    continue
                else:  # unit_reduce
                    adj[reduce_comp_to_pre[comp].readable_id].append(func.readable_id)
                    in_degree[func.readable_id] += 1
                    continue
            for port in comp.in_ports:
                if port.connected:
                    predecessor_comp = port.connection.parent
                    if predecessor_comp in comp_to_func:
                        # For ReduceComponent, comp_to_func correctly points to unit_reduce, which produces the output
                        adj[comp_to_func[predecessor_comp].readable_id].append(func.readable_id)
                        in_degree[func.readable_id] += 1

        queue = [fid for fid, degree in in_degree.items() if degree == 0]
        sorted_funcs = []
        while queue:
            func_id = queue.pop(0)
            sorted_funcs.append(id_to_func[func_id])
            for successor_id in adj[func_id]:
                in_degree[successor_id] -= 1
                if in_degree[successor_id] == 0:
                    queue.append(successor_id)

        if len(sorted_funcs) != len(stream_funcs):
            raise RuntimeError("A cycle was detected in the top-level dataflow graph.")

        # 4. Generate calls (using the user-provided, proven logic)
        body.append(CodeComment("--- Function Calls (in topological order) ---"))
        handled_unit_reduce_ids = set()
        for func in sorted_funcs:
            comp = func.dfir_comp
            if "unit_reduce" in func.name and comp.readable_id in handled_unit_reduce_ids:
                continue

            if "pre_process" in func.name and isinstance(comp, dfir.ReduceComponent):
                comp_id = comp.readable_id
                helpers = self.reduce_helpers[comp_id]
                streams = helpers["streams"]
                unit_reduce_func = helpers["unit_reduce"]

                body.append(CodeComment(f"--- Start of Reduce Super-Block for {comp.name} ---"))

                in_port = comp.get_port("i_0")
                pred_port = in_port.connection
                in_stream_var = (
                    top_io_map[in_port.readable_id]
                    if isinstance(pred_port.parent, dfir.IOComponent)
                    else stream_map[f"stream_{pred_port.unique_name}"]
                )

                pre_process_call_params = [
                    in_stream_var,
                    streams["intermediate_key"],
                    streams["intermediate_transform"],
                ]
                body.append(CodeCall(func, pre_process_call_params))

                body.append(
                    CodeCall(
                        helpers["zipper"],
                        [
                            streams["intermediate_key"],
                            streams["intermediate_transform"],
                            streams["zipper_to_demux"],
                        ],
                    )
                )
                body.append(
                    CodeCall(helpers["demux"], [streams["zipper_to_demux"], streams["demux_to_omega"]])
                )
                body.append(CodeCall(helpers["omega"], [streams["demux_to_omega"], streams["omega_to_unit"]]))

                out_port = unit_reduce_func.dfir_comp.get_port("o_0")
                out_stream_var = (
                    top_io_map[out_port.readable_id]
                    if out_port.connection is None
                    else stream_map[f"stream_{out_port.unique_name}"]
                )

                body.append(CodeCall(unit_reduce_func, [streams["omega_to_unit"], out_stream_var]))

                body.append(CodeComment(f"--- End of Reduce Super-Block for {comp.name} ---"))
                handled_unit_reduce_ids.add(comp_id)

            elif not isinstance(comp, dfir.ReduceComponent):
                call_params: List[HLSVar] = []
                for func_param in func.params:
                    port = comp.get_port(func_param.name)

                    if port.port_type == dfir.PortType.IN:
                        predecessor_port = port.connection
                        if isinstance(predecessor_port.parent, dfir.IOComponent):
                            call_params.append(top_io_map[port.readable_id])
                        else:
                            call_params.append(stream_map[f"stream_{predecessor_port.unique_name}"])
                    else:  # OUT port
                        if port.connection is None:
                            call_params.append(top_io_map[port.readable_id])
                        else:
                            call_params.append(stream_map[f"stream_{port.unique_name}"])
                body.append(CodeCall(func, call_params))

        return body

    def _generate_source_file(self, header_name: str, axi_wrapper_func_str: str) -> str:
        """Generates the full content of the .cpp source file with correct function order."""
        code = f'#include "{header_name}"\n\n'

        # --- *** 关键修正：调整函数定义顺序 *** ---
        # 顺序: 辅助网络 -> DFIR组件 -> AXI数据搬运 -> AXI顶层封装

        # 1. Utility Network Functions (callees)
        if self.utility_functions:
            code += "// --- Utility Network Functions ---\n"
            for func in self.utility_functions:
                params_str = ", ".join(
                    [p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT) for p in func.params]
                )
                code += f"void {func.name}({params_str}) " + "{\n"
                code += "".join([line.gen_code(1) for line in func.codes])
                code += "}\n\n"

        # 2. DFIR Component Functions (callees)
        code += "// --- DFIR Component Functions ---\n"
        for func in self.hls_functions.values():
            params_str = ", ".join([p.type.get_upper_param(p.name, True) for p in func.params])
            code += f"void {func.name}({params_str}) " + "{\n"
            code += "".join([line.gen_code(1) for line in func.codes])
            code += "}\n\n"

        # 3. AXI Helper Functions (callers)
        #    Note: The dataflow core function must come AFTER the components it calls.
        for func in [self.mem_to_stream_func, self.stream_to_mem_func, self.dataflow_core_func]:
            if func:
                params_str_list = []
                for p in func.params:
                    if p.type.type == HLSBasicType.POINTER:
                        params_str_list.append(p.type.get_upper_decl(p.name))
                    elif "uint16_t" in p.type.name:
                        params_str_list.append(f"uint16_t {p.name}")
                    else:
                        params_str_list.append(p.type.get_upper_param(p.name, True))

                params_str = ", ".join(params_str_list)
                code += f"static void {func.name}({params_str}) " + "{\n"
                code += "".join([line.gen_code(1) for line in func.codes])
                code += "}\n\n"

        # 4. Top-level AXI Kernel Wrapper (final caller)
        code += axi_wrapper_func_str

        return code

```

`CCFSys2025_GraphyFlow/graphyflow/backend_utils.py`:

```py
# backend_utils.py
from __future__ import annotations
import math
from typing import List

# 从您现有的后端和IR文件中导入必要的类
# 注意：我们现在从 backend_defines 导入基础类
import graphyflow.backend_defines as hls
import graphyflow.dataflow_ir as dfir

# 全局计数器，确保每次生成的函数名都唯一
_generator_id_counter = 0


def _get_unique_id() -> int:
    """获取一个唯一的ID用于函数命名。"""
    global _generator_id_counter
    id = _generator_id_counter
    _generator_id_counter += 1
    return id


def create_non_blocking_read(stream_var: hls.HLSVar, body_if_not_empty: List[hls.HLSCodeLine]) -> hls.CodeIf:
    """
    一个辅助函数，用于快速生成非阻塞读数据流的代码块。
    """
    empty_expr = hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, stream_var)])
    not_empty_expr = hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [empty_expr])
    # 修复了构造函数调用，确保默认参数安全
    return hls.CodeIf(expr=not_empty_expr, if_codes=body_if_not_empty)


# ======================================================================== #
#                      第二步：新增函数生成器                                #
# ======================================================================== #


def generate_merge_stream_2x1(data_type: hls.HLSType) -> hls.HLSFunction:
    """
    生成一个2合1数据流合并单元 (mergeStream2x1) 的 HLSFunction 对象。
    这个函数是构成归约树的基础。
    """
    gen_id = _get_unique_id()
    func_name = f"mergeStream2x1_{gen_id}"

    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[data_type])
    bool_type = hls.HLSType(hls.HLSBasicType.BOOL)
    int_type = hls.HLSType(hls.HLSBasicType.INT)

    # 定义函数参数
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("out", stream_type),
    ]

    # 定义函数体内的变量
    in1_end_flag_var = hls.HLSVar("in1_end_flag", bool_type)
    in2_end_flag_var = hls.HLSVar("in2_end_flag", bool_type)
    data1_var = hls.HLSVar("data1", data_type)
    data2_var = hls.HLSVar("data2", data_type)

    # 为了模拟 read_from_stream_nb，我们需要一个辅助函数来生成读取和检查的逻辑
    def create_nb_read_logic(in_stream_var, data_var, process_flag_var, end_flag_var):
        end_check_expr = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "end_flag"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )

        # if (data.end_flag) inX_end_flag = 1;
        set_end_flag = hls.CodeIf(
            end_check_expr, [hls.CodeAssign(end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))]
        )

        read_body = [
            hls.CodeAssign(process_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True)),
            hls.CodeVarDecl(data_var.name, data_var.type),
            hls.CodeAssign(
                data_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, in_stream_var)]),
            ),
            set_end_flag,
        ]

        return create_non_blocking_read(in_stream_var, read_body)

    # 构建主循环体
    p1_flag = hls.HLSVar("in1_process_flag", bool_type)
    p2_flag = hls.HLSVar("in2_process_flag", bool_type)

    # if(in1_process_flag && (!in1_end_flag)) write_to_stream(out, data1);
    write_cond1 = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, p1_flag),
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                dfir.UnaryOp.NOT,
                [hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var)],
            ),
        ],
    )
    write_block1 = hls.CodeIf(write_cond1, [hls.CodeWriteStream(params[3], data1_var)])

    write_cond2 = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, p2_flag),
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                dfir.UnaryOp.NOT,
                [hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var)],
            ),
        ],
    )
    write_block2 = hls.CodeIf(write_cond2, [hls.CodeWriteStream(params[3], data2_var)])

    # 退出逻辑
    exit_cond = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var),
            hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var),
        ],
    )
    end_data_var = hls.HLSVar("data", data_type)
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(params[3], end_data_var),
            hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeBreak(),
        ],
    )

    while_body = [
        hls.CodePragma("PIPELINE II=2"),
        # 初始化 process flags
        hls.CodeAssign(p1_flag, hls.HLSExpr(hls.HLSExprT.CONST, False)),
        hls.CodeAssign(p2_flag, hls.HLSExpr(hls.HLSExprT.CONST, False)),
        # 读取
        create_nb_read_logic(params[1], data1_var, p1_flag, in1_end_flag_var),
        create_nb_read_logic(params[2], data2_var, p2_flag, in2_end_flag_var),
        # 写入
        write_block1,
        write_block2,
        # 退出
        exit_block,
    ]

    func_body = [
        hls.CodePragma("function_instantiate variable=i"),
        hls.CodeVarDecl(in1_end_flag_var.name, in1_end_flag_var.type),
        hls.CodeVarDecl(in2_end_flag_var.name, in2_end_flag_var.type),
        hls.CodeVarDecl(p1_flag.name, p1_flag.type),
        hls.CodeVarDecl(p2_flag.name, p2_flag.type),
        hls.CodeWhile(codes=while_body, iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True)),
    ]

    merge_func = hls.HLSFunction(name=func_name, comp=None)
    merge_func.params = params
    merge_func.codes = func_body
    return merge_func


def generate_reduction_tree(n: int, data_type: hls.HLSType, merge_func: hls.HLSFunction) -> hls.HLSFunction:
    """
    生成一个 N->1 的归约树。
    """
    if not (n > 0 and (n & (n - 1) == 0)):
        raise ValueError("归约树的输入数量 'n' 必须是2的幂。")

    gen_id = _get_unique_id()
    func_name = f"reductionTree_{gen_id}"
    log_n = int(math.log2(n))

    # 定义函数参数
    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[data_type])
    stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[n])

    params = [
        hls.HLSVar("i", hls.HLSType(hls.HLSBasicType.INT)),
        # *** 接口统一：使用流数组 ***
        hls.HLSVar("update_set_stm", stream_array_type),
        hls.HLSVar("reduced_update_tuple_stm", stream_type),
    ]

    # 定义函数体
    body = [hls.CodePragma("DATAFLOW")]

    # 声明所有中间流
    streams_by_level = {}
    num_streams = n
    for level in range(log_n - 1):  # Only need log_n - 1 levels of intermediate streams
        num_streams //= 2
        if num_streams == 0:
            break
        level_stream_type = hls.HLSType(
            hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[num_streams]
        )
        level_stream_var = hls.HLSVar(f"l{level+1}_update_tuples", level_stream_type)
        streams_by_level[level + 1] = level_stream_var
        body.append(hls.CodeVarDecl(level_stream_var.name, level_stream_var.type))
        body.append(hls.CodePragma(f"STREAM variable={level_stream_var.name} depth=2"))

    # 生成 mergeStream2x1 调用
    num_mergers_at_level = n // 2
    input_streams = params[1]  # The initial array of streams
    for level in range(log_n):
        output_streams = streams_by_level.get(level + 1, params[2])  # Final output is the function param
        for i in range(num_mergers_at_level):
            in1_var = hls.HLSVar(f"{input_streams.name}[{i*2}]", stream_type)
            in2_var = hls.HLSVar(f"{input_streams.name}[{i*2 + 1}]", stream_type)

            # 如果是最后一级，输出是单个流，否则是数组中的一个元素
            if level == log_n - 1:
                out_var = output_streams
            else:
                out_var = hls.HLSVar(f"{output_streams.name}[{i}]", stream_type)

            call = hls.CodeCall(
                merge_func,
                [
                    hls.HLSExpr(hls.HLSExprT.CONST, i),  # 'i' for instantiation
                    in1_var,
                    in2_var,
                    out_var,
                ],
            )
            body.append(call)

        input_streams = output_streams
        num_mergers_at_level //= 2

    tree_func = hls.HLSFunction(name=func_name, comp=None)
    tree_func.params = params
    tree_func.codes = body
    return tree_func


def generate_demux(n: int, batch_type: hls.HLSType, wrapper_type: hls.HLSType) -> hls.HLSFunction:
    """
    生成一个 1->N 的数据流拆分器 (Demux/Unbatcher)。
    它接收批处理流，输出N个独立流。
    """
    gen_id = _get_unique_id()
    func_name = f"demux_{gen_id}"

    in_stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[batch_type])
    out_stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[wrapper_type])
    out_stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[out_stream_type], array_dims=[n])

    params = [
        hls.HLSVar("in_batch_stream", in_stream_type),
        hls.HLSVar("out_streams", out_stream_array_type),
    ]

    in_batch_var = hls.HLSVar("in_batch", batch_type)
    wrapper_var = hls.HLSVar("wrapper_data", wrapper_type)

    # --- 修正后的内部循环逻辑 ---
    # 仅当 `i < in_batch.end_pos` 时才分发数据
    inner_loop_body_if = [
        hls.CodeAssign(
            hls.HLSVar(f"{wrapper_var.name}.data", wrapper_type.sub_types[0]),
            hls.HLSExpr(
                hls.HLSExprT.VAR, hls.HLSVar(f"{in_batch_var.name}.data[i]", wrapper_type.sub_types[0])
            ),
        ),
        hls.CodeAssign(
            hls.HLSVar(f"{wrapper_var.name}.end_flag", hls.HLSType(hls.HLSBasicType.BOOL)),
            hls.HLSExpr(hls.HLSExprT.CONST, False),
        ),
        hls.CodeWriteStream(hls.HLSVar(f"out_streams[i]", out_stream_type), wrapper_var),
    ]

    cond_expr = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.LT,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar("i", hls.HLSType(hls.HLSBasicType.UINT))),
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "end_pos"),
                [hls.HLSExpr(hls.HLSExprT.VAR, in_batch_var)],
            ),
        ],
    )

    inner_for_loop = hls.CodeFor(
        [hls.CodePragma("UNROLL"), hls.CodeIf(cond_expr, inner_loop_body_if)],
        iter_limit="PE_NUM",
        iter_name="i",
    )

    # While 循环体
    while_body = [
        hls.CodePragma("PIPELINE"),
        hls.CodeAssign(
            in_batch_var,
            hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[0])]),
        ),
        hls.CodeVarDecl(wrapper_var.name, wrapper_type),
        inner_for_loop,
        hls.CodeIf(
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "end_flag"),
                [hls.HLSExpr(hls.HLSExprT.VAR, in_batch_var)],
            ),
            [hls.CodeBreak()],
        ),
    ]

    # 结束标志传播逻辑
    end_wrapper_var = hls.HLSVar("end_wrapper", wrapper_type)
    final_loop = hls.CodeFor(
        [
            hls.CodePragma("UNROLL"),
            hls.CodeWriteStream(hls.HLSVar(f"out_streams[i]", out_stream_type), end_wrapper_var),
        ],
        iter_limit=n,
        iter_name="i",
    )

    body = [
        hls.CodeVarDecl(in_batch_var.name, in_batch_var.type),
        hls.CodeWhile(codes=while_body, iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True)),
        hls.CodeComment("Propagate end_flag to all output streams"),
        hls.CodeVarDecl(end_wrapper_var.name, end_wrapper_var.type),
        hls.CodeAssign(
            hls.HLSVar(f"{end_wrapper_var.name}.end_flag", hls.HLSType(hls.HLSBasicType.BOOL)),
            hls.HLSExpr(hls.HLSExprT.CONST, True),
        ),
        final_loop,
    ]

    demux_func = hls.HLSFunction(name=func_name, comp=None)
    demux_func.params = params
    demux_func.codes = body
    return demux_func


# ======================================================================== #
#                      OMEGA NETWORK GENERATOR (FULL)                      #
# ======================================================================== #


def generate_omega_network(n: int, wrapper_type, routing_key_member: str) -> List[hls.HLSFunction]:
    """
    生成一个 N x N Omega 网络的完整HLS C++代码，包括所有子模块。
    """
    if not (n > 0 and (n & (n - 1) == 0)):
        raise ValueError("网络大小 'n' 必须是2的正整数次幂。")

    log_n = int(math.log2(n))
    switches_per_stage = n // 2
    gen_id = _get_unique_id()

    # --- HLS类型定义 ---

    # b. 其他类型现在基于 wrapper_type
    stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[wrapper_type])
    bool_type = hls.HLSType(hls.HLSBasicType.BOOL)
    int_type = hls.HLSType(hls.HLSBasicType.INT)
    stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[n])

    # --- 调用子模块生成器 ---
    sender_function = generate_omega_sender(
        gen_id, wrapper_type, stream_type, int_type, bool_type, routing_key_member
    )
    receiver_function = generate_omega_receiver(gen_id, wrapper_type, stream_type, int_type, bool_type)
    switch2x2_function = generate_omega_switch2x2(
        gen_id, stream_type, int_type, sender_function, receiver_function
    )
    omega_switch_function = generate_omega_top(
        gen_id, n, log_n, n // 2, stream_array_type, stream_type, switch2x2_function
    )

    return [sender_function, receiver_function, switch2x2_function, omega_switch_function]


# Helper functions to keep generate_omega_network clean
def generate_omega_sender(gen_id, data_tuple_type, stream_type, int_type, bool_type, routing_key_member: str):
    func_name = f"sender_{gen_id}"
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("out1", stream_type),
        hls.HLSVar("out2", stream_type),
        hls.HLSVar("out3", stream_type),
        hls.HLSVar("out4", stream_type),
    ]
    in1_end_flag_var = hls.HLSVar("in1_end_flag", bool_type)
    in2_end_flag_var = hls.HLSVar("in2_end_flag", bool_type)
    data1_var, data2_var = hls.HLSVar("data1", data_tuple_type), hls.HLSVar("data2", data_tuple_type)

    def create_routing_expr(data_var):
        inner_data_expr = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "data"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )
        key_expr = hls.HLSExpr(
            hls.HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, routing_key_member), [inner_data_expr]
        )

        i_var_expr = hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar("i", int_type))
        shifted_expr = hls.HLSExpr(hls.HLSExprT.BINOP, dfir.BinOp.SR, [key_expr, i_var_expr])
        return hls.HLSExpr(
            hls.HLSExprT.BINOP, dfir.BinOp.AND, [shifted_expr, hls.HLSExpr(hls.HLSExprT.CONST, 1)]
        )

    # Logic for in1
    end_check1 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, data1_var)],
    )
    route_if1 = hls.CodeIf(
        create_routing_expr(data1_var),
        if_codes=[hls.CodeWriteStream(params[4], data1_var)],
        else_codes=[hls.CodeWriteStream(params[3], data1_var)],
    )
    process_if1 = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check1]),
        if_codes=[route_if1],
        else_codes=[hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
    )
    nb_read1 = create_non_blocking_read(
        params[1],
        [
            hls.CodeVarDecl(data1_var.name, data1_var.type),
            hls.CodeAssign(
                data1_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[1])]),
            ),
            process_if1,
        ],
    )

    # Logic for in2
    end_check2 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, data2_var)],
    )
    route_if2 = hls.CodeIf(
        create_routing_expr(data2_var),
        if_codes=[hls.CodeWriteStream(params[6], data2_var)],
        else_codes=[hls.CodeWriteStream(params[5], data2_var)],
    )
    process_if2 = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check2]),
        if_codes=[route_if2],
        else_codes=[hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
    )
    nb_read2 = create_non_blocking_read(
        params[2],
        [
            hls.CodeVarDecl(data2_var.name, data2_var.type),
            hls.CodeAssign(
                data2_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[2])]),
            ),
            process_if2,
        ],
    )

    # Exit logic
    exit_cond = hls.HLSExpr(
        hls.HLSExprT.BINOP,
        dfir.BinOp.AND,
        [
            hls.HLSExpr(hls.HLSExprT.VAR, in1_end_flag_var),
            hls.HLSExpr(hls.HLSExprT.VAR, in2_end_flag_var),
        ],
    )
    end_data_var = hls.HLSVar("data", data_tuple_type)
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(params[3], end_data_var),
            hls.CodeWriteStream(params[4], end_data_var),
            hls.CodeWriteStream(params[5], end_data_var),
            hls.CodeWriteStream(params[6], end_data_var),
            hls.CodeAssign(in1_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeAssign(in2_end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
            hls.CodeBreak(),
        ],
    )

    body = [
        hls.CodePragma("function_instantiate variable=i"),
        hls.CodeVarDecl(in1_end_flag_var.name, in1_end_flag_var.type, init_val="false"),
        hls.CodeVarDecl(in2_end_flag_var.name, in2_end_flag_var.type, init_val="false"),
        hls.CodeWhile(
            [hls.CodePragma("PIPELINE II=1"), nb_read1, nb_read2, exit_block],
            hls.HLSExpr(hls.HLSExprT.CONST, True),
        ),
    ]
    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_omega_receiver(gen_id, data_tuple_type, stream_type, int_type, bool_type):
    func_name = f"receiver_{gen_id}"
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("out1", stream_type),
        hls.HLSVar("out2", stream_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("in3", stream_type),
        hls.HLSVar("in4", stream_type),
    ]
    end_flags = [hls.HLSVar(f"in{i+1}_end_flag", bool_type) for i in range(4)]

    def get_read_body(in_stream, out_stream, flag_var):
        data_var = hls.HLSVar("data", data_tuple_type)
        end_check = hls.HLSExpr(
            hls.HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "end_flag"),
            [hls.HLSExpr(hls.HLSExprT.VAR, data_var)],
        )
        process_if = hls.CodeIf(
            hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check]),
            if_codes=[hls.CodeWriteStream(out_stream, data_var)],
            else_codes=[hls.CodeAssign(flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
        )
        return [
            hls.CodeVarDecl(data_var.name, data_var.type),
            hls.CodeAssign(
                data_var,
                hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, in_stream)]),
            ),
            process_if,
        ]

    # Merge logic
    cond1 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[3])])],
    )
    cond3 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[5])])],
    )
    merge13 = hls.CodeIf(
        cond1,
        get_read_body(params[3], params[1], end_flags[0]),
        elifs=[(cond3, get_read_body(params[5], params[1], end_flags[2]))],
    )

    cond2 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[4])])],
    )
    cond4 = hls.HLSExpr(
        hls.HLSExprT.UOP,
        dfir.UnaryOp.NOT,
        [hls.HLSExpr(hls.HLSExprT.STREAM_EMPTY, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[6])])],
    )
    merge24 = hls.CodeIf(
        cond2,
        get_read_body(params[4], params[2], end_flags[1]),
        elifs=[(cond4, get_read_body(params[6], params[2], end_flags[3]))],
    )

    # Exit logic
    exit_cond = hls.HLSExpr(hls.HLSExprT.VAR, end_flags[0])
    for i in range(1, 4):
        exit_cond = hls.HLSExpr(
            hls.HLSExprT.BINOP,
            dfir.BinOp.AND,
            [exit_cond, hls.HLSExpr(hls.HLSExprT.VAR, end_flags[i])],
        )
    end_data_var = hls.HLSVar("data", data_tuple_type)
    exit_block = hls.CodeIf(
        exit_cond,
        [
            hls.CodeVarDecl(end_data_var.name, end_data_var.type),
            hls.CodeAssign(
                hls.HLSVar(f"{end_data_var.name}.end_flag", bool_type),
                hls.HLSExpr(hls.HLSExprT.CONST, True),
            ),
            hls.CodeWriteStream(params[1], end_data_var),
            hls.CodeWriteStream(params[2], end_data_var),
            hls.CodeBreak(),
        ],
    )

    body = [hls.CodePragma("function_instantiate variable=i")]
    body.extend([hls.CodeVarDecl(v.name, v.type, init_val="false") for v in end_flags])
    body.append(
        hls.CodeWhile(
            iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True),
            codes=[hls.CodePragma("PIPELINE II=1"), merge13, merge24, exit_block],
        )
    )

    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_omega_switch2x2(gen_id, stream_type, int_type, sender_func, receiver_func):
    func_name = f"switch2x2_{gen_id}"
    params = [
        hls.HLSVar("i", int_type),
        hls.HLSVar("in1", stream_type),
        hls.HLSVar("in2", stream_type),
        hls.HLSVar("out1", stream_type),
        hls.HLSVar("out2", stream_type),
    ]
    local_streams = [hls.HLSVar(f"l1_{i+1}", stream_type) for i in range(4)]
    body = [hls.CodePragma("DATAFLOW")]
    for stream_var in local_streams:
        body.append(hls.CodeVarDecl(stream_var.name, stream_var.type))
        body.append(hls.CodePragma(f"STREAM variable={stream_var.name} depth=2"))

    sender_call = hls.CodeCall(
        sender_func,
        [
            params[0],
            params[1],
            params[2],
            local_streams[0],
            local_streams[1],
            local_streams[2],
            local_streams[3],
        ],
    )
    receiver_call = hls.CodeCall(
        receiver_func,
        [
            params[0],
            params[3],
            params[4],
            local_streams[0],
            local_streams[1],
            local_streams[2],
            local_streams[3],
        ],
    )
    body.extend([sender_call, receiver_call])

    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_omega_top(gen_id, n, log_n, switches_per_stage, stream_array_type, stream_type, switch2x2_func):
    func_name = f"omega_switch_{gen_id}"
    params = [
        hls.HLSVar("in_streams", stream_array_type),
        hls.HLSVar("out_streams", stream_array_type),
    ]
    body = [hls.CodePragma("DATAFLOW")]

    intermediate_streams = []
    for s in range(log_n - 1):
        stage_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[n])
        stage_var = hls.HLSVar(f"stream_stage_{s}", stage_array_type)
        intermediate_streams.append(stage_var)
        body.append(hls.CodeVarDecl(stage_var.name, stage_var.type))
        body.append(hls.CodePragma(f"STREAM variable={stage_var.name} depth=2"))

    def unshuffle(p, num_bits):
        return ((p & 1) << (num_bits - 1)) | (p >> 1)

    for s in range(log_n):
        for j in range(switches_per_stage):
            idx1, idx2 = 2 * j, 2 * j + 1
            if s == 0:
                in1_var, in2_var = hls.HLSVar(f"in_streams[{idx1}]", stream_type), hls.HLSVar(
                    f"in_streams[{idx2}]", stream_type
                )
            else:
                uidx1, uidx2 = unshuffle(idx1, log_n), unshuffle(idx2, log_n)
                in_array = intermediate_streams[s - 1]
                in1_var, in2_var = hls.HLSVar(f"{in_array.name}[{uidx1}]", stream_type), hls.HLSVar(
                    f"{in_array.name}[{uidx2}]", stream_type
                )

            if s == log_n - 1:
                out1_var, out2_var = hls.HLSVar(f"out_streams[{idx1}]", stream_type), hls.HLSVar(
                    f"out_streams[{idx2}]", stream_type
                )
            else:
                out_array = intermediate_streams[s]
                out1_var, out2_var = hls.HLSVar(f"{out_array.name}[{idx1}]", stream_type), hls.HLSVar(
                    f"{out_array.name}[{idx2}]", stream_type
                )

            call = hls.CodeCall(
                switch2x2_func,
                [
                    hls.HLSExpr(hls.HLSExprT.CONST, log_n - s - 1),
                    in1_var,
                    in2_var,
                    out1_var,
                    out2_var,
                ],
            )
            body.append(call)

    func = hls.HLSFunction(name=func_name, comp=None)
    func.params, func.codes = params, body
    return func


def generate_stream_zipper(
    key_batch_type: hls.HLSType,
    transform_batch_type: hls.HLSType,
    out_kt_pair_batch_type: hls.HLSType,
) -> hls.HLSFunction:
    """
    生成一个流合并器 (Stream Zipper)。
    输入: 两个批处理流 (key, transform)
    输出: 一个合并了key和transform的批处理流 (kt_pair)
    """
    gen_id = _get_unique_id()
    func_name = f"stream_zipper_{gen_id}"

    params = [
        hls.HLSVar("in_key_batch_stream", hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[key_batch_type])),
        hls.HLSVar(
            "in_transform_batch_stream",
            hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[transform_batch_type]),
        ),
        hls.HLSVar(
            "out_pair_batch_stream",
            hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[out_kt_pair_batch_type]),
        ),
    ]

    key_batch_var = hls.HLSVar("key_batch", key_batch_type)
    transform_batch_var = hls.HLSVar("transform_batch", transform_batch_type)
    out_batch_var = hls.HLSVar("out_batch", out_kt_pair_batch_type)

    # 内部循环: out.data[i].key = key.data[i]; out.data[i].transform = transform.data[i];
    kt_pair_type = out_kt_pair_batch_type.sub_types[0].sub_types[0]
    assign_key = hls.CodeAssign(
        hls.HLSVar(f"{out_batch_var.name}.data[i].key", kt_pair_type.sub_types[0]),
        hls.HLSVar(f"{key_batch_var.name}.data[i]", key_batch_type.sub_types[0].sub_types[0]),
    )
    assign_transform = hls.CodeAssign(
        hls.HLSVar(f"{out_batch_var.name}.data[i].transform", kt_pair_type.sub_types[1]),
        hls.HLSVar(f"{transform_batch_var.name}.data[i]", transform_batch_type.sub_types[0].sub_types[0]),
    )
    for_loop = hls.CodeFor(
        [hls.CodePragma("UNROLL"), assign_key, assign_transform],
        iter_limit="PE_NUM",
        iter_name="i",
    )

    # while循环体
    end_flag_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, key_batch_var)],
    )
    end_pos_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_pos"),
        [hls.HLSExpr(hls.HLSExprT.VAR, key_batch_var)],
    )

    while_body = [
        hls.CodePragma("PIPELINE"),
        hls.CodeAssign(
            key_batch_var,
            hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[0])]),
        ),
        hls.CodeAssign(
            transform_batch_var,
            hls.HLSExpr(hls.HLSExprT.STREAM_READ, None, [hls.HLSExpr(hls.HLSExprT.VAR, params[1])]),
        ),
        for_loop,
        hls.CodeAssign(
            hls.HLSVar(f"{out_batch_var.name}.end_flag", hls.HLSType(hls.HLSBasicType.BOOL)),
            end_flag_expr,
        ),
        hls.CodeAssign(
            hls.HLSVar(f"{out_batch_var.name}.end_pos", hls.HLSType(hls.HLSBasicType.UINT8)),
            end_pos_expr,
        ),
        hls.CodeWriteStream(params[2], out_batch_var),
        hls.CodeIf(end_flag_expr, [hls.CodeBreak()]),
    ]

    body = [
        hls.CodeVarDecl(key_batch_var.name, key_batch_var.type),
        hls.CodeVarDecl(transform_batch_var.name, transform_batch_var.type),
        hls.CodeVarDecl(out_batch_var.name, out_batch_var.type),
        hls.CodeWhile(iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True), codes=while_body),
    ]

    zipper_func = hls.HLSFunction(name=func_name, comp=None)
    zipper_func.params = params
    zipper_func.codes = body
    return zipper_func


def generate_stream_unzipper(
    n: int,
    wrapped_kt_pair_type: hls.HLSType,
    out_key_stream_array_type: hls.HLSType,
    out_transform_stream_array_type: hls.HLSType,
) -> hls.HLSFunction:
    """
    生成一个流拆分器 (Stream Unzipper)。
    输入: 一个流数组 (承载 wrapper<kt_pair>)
    输出: 两个流数组 (key, transform)
    """
    gen_id = _get_unique_id()
    func_name = f"stream_unzipper_{gen_id}"

    in_stream_type = hls.HLSType(hls.HLSBasicType.STREAM, sub_types=[wrapped_kt_pair_type])
    in_stream_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[in_stream_type], array_dims=[n])

    params = [
        hls.HLSVar("in_streams", in_stream_array_type),
        hls.HLSVar("out_key_streams", out_key_stream_array_type),
        hls.HLSVar("out_transform_streams", out_transform_stream_array_type),
    ]

    wrapper_var = hls.HLSVar("wrapper", wrapped_kt_pair_type)
    end_flag_var = hls.HLSVar(
        "end_flag_local",
        hls.HLSType(hls.HLSBasicType.BOOL),
    )

    # 内部循环体
    end_check_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "end_flag"),
        [hls.HLSExpr(hls.HLSExprT.VAR, wrapper_var)],
    )

    # 表达式: wrapper.data.key
    key_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "key"),
        [
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "data"),
                [hls.HLSExpr(hls.HLSExprT.VAR, wrapper_var)],
            )
        ],
    )
    # 表达式: wrapper.data.transform
    transform_expr = hls.HLSExpr(
        hls.HLSExprT.UOP,
        (dfir.UnaryOp.GET_ATTR, "transform"),
        [
            hls.HLSExpr(
                hls.HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "data"),
                [hls.HLSExpr(hls.HLSExprT.VAR, wrapper_var)],
            )
        ],
    )

    write_key = hls.CodeWriteStream(
        hls.HLSVar(f"out_key_streams[i]", out_key_stream_array_type.sub_types[0]), key_expr
    )
    write_transform = hls.CodeWriteStream(
        hls.HLSVar(f"out_transform_streams[i]", out_transform_stream_array_type.sub_types[0]),
        transform_expr,
    )

    unzip_if_block = hls.CodeIf(
        hls.HLSExpr(hls.HLSExprT.UOP, dfir.UnaryOp.NOT, [end_check_expr]),
        if_codes=[write_key, write_transform],
        else_codes=[hls.CodeAssign(end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, True))],
    )

    # For循环
    for_loop_body = [
        hls.CodePragma("UNROLL"),
        hls.CodeAssign(
            wrapper_var,
            hls.HLSExpr(
                hls.HLSExprT.STREAM_READ,
                None,
                [hls.HLSExpr(hls.HLSExprT.VAR, hls.HLSVar(f"in_streams[i]", in_stream_type))],
            ),
        ),
        unzip_if_block,
    ]
    for_loop = hls.CodeFor(for_loop_body, iter_limit=n, iter_name="i")

    # While循环
    while_body = [
        hls.CodePragma("PIPELINE"),
        hls.CodeAssign(end_flag_var, hls.HLSExpr(hls.HLSExprT.CONST, False)),
        hls.CodeVarDecl(wrapper_var.name, wrapper_var.type),
        for_loop,
        hls.CodeIf(end_flag_var, [hls.CodeBreak()]),
    ]

    body = [
        hls.CodeVarDecl(end_flag_var.name, end_flag_var.type, init_val="false"),
        hls.CodeWhile(iter_expr=hls.HLSExpr(hls.HLSExprT.CONST, True), codes=while_body),
    ]

    unzipper_func = hls.HLSFunction(name=func_name, comp=None)
    unzipper_func.params = params
    unzipper_func.codes = body
    return unzipper_func


# ======================================================================== #
#                      测试代码入口                                        #
# ======================================================================== #
if __name__ == "__main__":

    def print_hls_function(func: hls.HLSFunction):
        """辅助函数，用于打印单个HLSFunction对象的C++代码。"""
        # 修正函数签名的生成，以正确处理流数组
        params_list = []
        for p in func.params:
            if p.type.type == hls.HLSBasicType.ARRAY and p.type.sub_types[0].type == hls.HLSBasicType.STREAM:
                # C++ 数组参数语法: type name[]
                base_type_str = p.type.sub_types[0].name
                params_list.append(f"{base_type_str} {p.name}[]")
            else:
                # 普通参数语法: type& name
                params_list.append(f"{p.type.name}& {p.name}")

        params_str = ", ".join(params_list)

        print(f"// --- Function: {func.name} ---")
        print(f"void {func.name}({params_str}) {{")
        code_body = "".join([line.gen_code(indent_lvl=1) for line in func.codes])
        print(code_body, end="")
        print("}\n")

    print("=" * 50)
    print("      Running backend_utils.py Test Suite")
    print("=" * 50)

    # --- 1. 定义测试用的数据类型 ---
    N = 8
    DATA_TYPE_NAME = "update_tuple_dt"

    print(f"\n[INFO] Using N={N} and data_type='{DATA_TYPE_NAME}' for tests.\n")

    # a. 基础数据类型
    data_type = hls.HLSType(
        hls.HLSBasicType.STRUCT,
        sub_types=[hls.HLSType(hls.HLSBasicType.UINT), hls.HLSType(hls.HLSBasicType.BOOL)],
        struct_name=DATA_TYPE_NAME,
        struct_prop_names=["dst", "end_flag"],
    )
    # b. 批处理类型 (用于 Demux 测试)
    batch_data_array_type = hls.HLSType(hls.HLSBasicType.ARRAY, sub_types=[data_type], array_dims=[N])
    batch_type = hls.HLSType(
        hls.HLSBasicType.STRUCT,
        sub_types=[
            batch_data_array_type,
            hls.HLSType(hls.HLSBasicType.BOOL),
            hls.HLSType(hls.HLSBasicType.UINT8),
        ],
        struct_prop_names=["data", "end_flag", "end_pos"],
    )

    # --- 2. 测试 generate_merge_stream_2x1 ---
    print("=" * 20, "Test: generate_merge_stream_2x1", "=" * 20)
    merge_func = generate_merge_stream_2x1(data_type)
    print_hls_function(merge_func)

    # --- 3. 测试 generate_reduction_tree ---
    print("=" * 20, "Test: generate_reduction_tree", "=" * 22)
    tree_func = generate_reduction_tree(n=N, data_type=data_type, merge_func=merge_func)
    print_hls_function(tree_func)

    # --- 4. 测试 generate_demux ---
    print("=" * 20, "Test: generate_demux", "=" * 28)
    demux_func = generate_demux(n=N, batch_type=batch_type)
    print_hls_function(demux_func)

    # --- 5. 测试 generate_omega_network ---
    print("=" * 20, "Test: generate_omega_network", "=" * 22)
    omega_functions = generate_omega_network(n=N, data_type_name=DATA_TYPE_NAME)
    for func in omega_functions:
        print_hls_function(func)

    print("=" * 50)
    print("            Test Suite Finished")
    print("=" * 50)

```

`CCFSys2025_GraphyFlow/graphyflow/dataflow_ir.py`:

```py
from __future__ import annotations
import uuid as uuid_lib
import copy
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
from graphyflow.dataflow_ir_datatype import *
import graphyflow.hls_utils as hls


class DfirNode:
    _readable_id = 0

    def __init__(self) -> None:
        self.uuid = uuid_lib.uuid4()
        self.readable_id = DfirNode._readable_id
        DfirNode._readable_id += 1


class EmptyNode(DfirNode):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


class PortType(Enum):
    IN = "in"
    OUT = "out"

    def pluggable(self, other: PortType) -> bool:
        return self != other

    def __eq__(self, other: PortType) -> bool:
        return self.value == other.value


class Port(DfirNode):
    def __init__(self, name: str, parent: DfirNode) -> None:
        super().__init__()
        self.name = name
        self.unique_name = f"{self.name}_{self.readable_id}"
        self.port_type = PortType.OUT if name.startswith("o_") else PortType.IN
        self.data_type = parent.input_type if self.port_type == PortType.IN else parent.output_type
        self.parent = parent
        self.connection = None

    def __eq__(self, other: Port):
        return (
            self.unique_name == other.unique_name
            and str(self.parent) == str(other.parent)
            and str(self) == str(other)
        )

    def __hash__(self) -> int:
        return hash(str(self) + self.unique_name)

    @property
    def connected(self) -> bool:
        return self.connection is not None

    @property
    def from_const(self) -> bool:
        assert self.port_type == PortType.IN and self.connected
        return isinstance(self.connection.parent, ConstantComponent)

    @property
    def from_const_val(self):
        assert self.from_const
        return self.connection.parent.value

    def copy(self, copy_comp: CopyComponent) -> Port:
        assert self.port_type == PortType.OUT
        assert self.connected
        assert self.data_type == copy_comp.input_type
        assert all(not p.connected for p in copy_comp.ports)
        original_connection = self.connection
        self.disconnect()
        copy_comp.connect({"i_0": self, "o_0": original_connection})
        return copy_comp.out_ports[1]

    def connect(self, other: Port) -> None:
        assert not self.connected and not other.connected
        assert self.port_type.pluggable(other.port_type)
        self.connection = other
        other.connection = self

    def disconnect(self) -> None:
        assert self.connection is not None
        self.connection.connection = None
        self.connection = None

    def __repr__(self) -> str:
        my_repr = f"Port[{self.readable_id}] {self.name} ({self.data_type})"
        direction = "=>" if self.port_type == PortType.OUT else "<="
        tgt = self.connection
        if self.connection is None:
            return my_repr
        else:
            return f"{my_repr} {direction} [{tgt.readable_id}] {tgt.name} ({tgt.data_type})"


class ComponentCollection(DfirNode):
    def __init__(self, components: List[Component], inputs: List[Port], outputs: List[Port]) -> None:
        super().__init__()
        self.components = components
        self.inputs = inputs
        self.outputs = outputs
        in_and_out = inputs + outputs
        assert all(all(p.connected or p in in_and_out for p in c.ports) for c in self.components)

    def __repr__(self) -> str:
        return f"ComponentCollection(\n  components: {self.components},\n  inputs: {self.inputs},\n  outputs: {self.outputs}\n)"

    @property
    def all_connected_ports(self) -> List[Port]:
        return [p for p in sum([comp.ports for comp in self.components], []) if p.connected]

    @property
    def output_types(self) -> List[DfirType]:
        return [p.data_type for p in self.outputs]

    def added(self, component: Component) -> bool:
        return component.readable_id in [c.readable_id for c in self.components]

    def update_ports(self) -> None:
        def remove_dup(ls: List[Port]) -> List[Port]:
            ls2 = []
            for p in ls:
                if p not in ls2:
                    ls2.append(p)
            return ls2

        # delete all connected ports in inputs and outputs, and delete replaced ports
        self.inputs = remove_dup([p for p in self.inputs if not p.connected])
        self.outputs = remove_dup([p for p in self.outputs if not p.connected])

    def add_front(self, component: Component, ports: Dict[str, Port]) -> None:
        assert all(p in self.inputs for p in ports.values())
        assert all(not p.connected or p in self.all_connected_ports for p in component.in_ports)
        component.connect(ports)
        if not self.added(component):
            self.components.insert(0, component)
        self.inputs = [p for p in self.inputs if p not in ports.values()]
        self.inputs.extend([p for p in component.in_ports])
        self.outputs.extend([p for p in component.out_ports])
        self.update_ports()

    def add_back(self, component: Component, ports: Dict[str, Port]) -> None:
        assert all(p in self.outputs for p in ports.values())
        assert all(not p.connected or p in self.all_connected_ports for p in component.out_ports)
        component.connect(ports)
        if not self.added(component):
            self.components.append(component)
        self.outputs = [p for p in self.outputs if p not in ports.values()]
        self.outputs.extend([p for p in component.out_ports])
        self.inputs.extend([p for p in component.in_ports])
        self.update_ports()

    def concat(
        self, other: ComponentCollection, port_connections: List[Tuple[Port, Port]]
    ) -> ComponentCollection:
        assert all(p in (self.inputs + self.outputs) for p, _ in port_connections)
        assert all(p in (other.inputs + other.outputs) for _, p in port_connections)
        for p, q in port_connections:
            p.connect(q)
        self.components.extend(other.components)
        for p in other.inputs:
            if p not in self.inputs:
                self.inputs.append(p)
        for p in other.outputs:
            if p not in self.outputs:
                self.outputs.append(p)
        for p, q in port_connections:
            for port in [p, q]:
                while port in self.inputs:
                    self.inputs.remove(port)
                while port in self.outputs:
                    self.outputs.remove(port)
        self.update_ports()
        return self

    def topo_sort(self) -> List[Component]:
        def port_solved(port: Port) -> bool:
            if not port.connected:
                assert port in (self.inputs + self.outputs)
                return True
            else:
                return port.connection.parent in result

        def check_reduce(comp: Component) -> bool:
            if not isinstance(comp, ReduceComponent):
                return False
            return port_solved(comp.get_port("i_0"))

        result = []
        waitings = copy.deepcopy(self.components)
        while waitings:
            new_ones = []
            for comp in waitings:
                if all(port_solved(p) for p in comp.in_ports) or check_reduce(comp):
                    new_ones.append(comp)
            waitings = [w for w in waitings if w not in new_ones]
            result.extend(new_ones)
        return result


class Component(DfirNode):
    def __init__(
        self,
        input_type: DfirType,
        output_type: DfirType,
        ports: List[str],
        parallel: bool = False,
        specific_port_types: Optional[Dict[str, DfirType]] = None,
    ) -> None:
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.ports = [Port(port, self) for port in ports]
        self.in_ports = [p for p in self.ports if p.port_type == PortType.IN]
        self.input_port_num = len(self.in_ports)
        self.out_ports = [p for p in self.ports if p.port_type == PortType.OUT]
        self.output_port_num = len(self.out_ports)
        self.parallel = parallel
        if specific_port_types is not None:
            for port_name, data_type in specific_port_types.items():
                for port in self.ports:
                    if port.name == port_name:
                        port.data_type = data_type
                        break
                else:
                    raise ValueError(f"Port {port_name} not found in {self.ports}")

    def get_port(self, name: str) -> Port:
        for port in self.ports:
            if port.name == name:
                return port
        raise ValueError(f"Port {name} not found in {self.ports}")

    def connect(self, ports: Union[List[Port], Dict[str, Port]]) -> None:
        if isinstance(ports, list):
            for i, port in enumerate(ports):
                assert port.data_type == self.ports[i].data_type
                self.ports[i].connect(port)
        elif isinstance(ports, dict):
            for port_name, port in ports.items():
                idx = None
                for i, p in enumerate(self.ports):
                    if p.name == port_name:
                        idx = i
                        break
                assert idx is not None
                assert (
                    port.data_type == self.ports[idx].data_type
                ), f"{port.data_type} != {self.ports[idx].data_type}"
                self.ports[idx].connect(port)

    def additional_info(self) -> str:
        return ""

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__[:5]}_{self.readable_id}"

    def to_hls(self) -> hls.HLSFunction:
        assert False, f"Abstract method to_hls() should be implemented for {self.__class__.__name__}"

    def get_hls_function(
        self,
        code_in_loop: List[str],
        code_before_loop: Optional[List[str]] = [],
        code_after_loop: Optional[List[str]] = [],
        name_tail: Optional[str] = None,
    ) -> hls.HLSFunction:
        return hls.HLSFunction(
            name=self.name + (f"_{name_tail}" if name_tail else ""),
            comp=self,
            code_in_loop=code_in_loop,
            code_before_loop=code_before_loop,
            code_after_loop=code_after_loop,
        )

    def __repr__(self) -> str:
        add_info = self.additional_info()
        add_info = add_info if isinstance(add_info, list) else [add_info]
        add_info = "\n  ".join(add_info)
        add_info = "\n  " + add_info if add_info else ""
        return (
            f"{self.__class__.__name__}(\n  input: {self.input_type},\n"
            + f"  output: {self.output_type},\n  ports:\n    "
            + "\n    ".join([str(p) for p in self.ports])
            + add_info
            + "\n)"
        )


class IOComponent(Component):
    class IOType(Enum):
        INPUT = "input"
        OUTPUT = "output"

    def __init__(self, io_type: IOType, data_type: DfirType) -> None:
        self.io_type = io_type
        if self.io_type == self.IOType.INPUT:
            super().__init__(None, data_type, ["o_0"])
        else:
            super().__init__(data_type, None, ["i_0"])

    def to_hls(self) -> hls.HLSFunction:
        assert False, "IOComponent should not be used in HLS"


class ConstantComponent(Component):
    def __init__(self, data_type: DfirType, value: Any) -> None:
        super().__init__(None, data_type, ["o_0"], parallel=isinstance(data_type, ArrayType))
        self.value = value

    def additional_info(self) -> str:
        return f"value: {self.value}"

    def to_hls(self) -> hls.HLSFunction:
        assert False, "ConstantComponent should not be used in HLS"


class CopyComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, input_type, ["i_0", "o_0", "o_1"])

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = [
            r"#type:i_0# copy_src = #read:i_0#;",
            r"bool #end_flag_val# = copy_src.end_flag;",
            # r"o_0.write(copy_src);",
            # r"o_1.write(copy_src);",
            r"#write_notrans:o_0,copy_src#",
            r"#write_notrans:o_1,copy_src#",
        ]
        return self.get_hls_function(code_in_loop)


class GatherComponent(Component):
    def __init__(self, input_types: List[DfirType]) -> None:
        parallel = all(isinstance(t, ArrayType) for t in input_types)
        ports = []
        specific_port_types = {}
        output_types = []
        for i in range(len(input_types)):
            ports.append(f"i_{i}")
            specific_port_types[f"i_{i}"] = input_types[i]
            if parallel:
                output_types.append(input_types[i].type_)
            else:
                output_types.append(input_types[i])
        output_type = TupleType(output_types)
        if parallel:
            output_type = ArrayType(output_type)
        ports.append("o_0")
        super().__init__(output_type, output_type, ports, parallel, specific_port_types)

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = ["bool #end_flag_val# = false;"]
        for i in range(len(self.in_ports)):
            code_in_loop.extend(
                [
                    f"#type:{self.in_ports[i].name}# gather_src_{i} = #read:i_{i}#;",
                    f"#end_flag_val# |= gather_src_{i}.end_flag;",
                    f"#peel:{self.in_ports[i].name},gather_src_{i},real_gather_src_{i}#",
                ]
            )
        code_in_loop += [
            r"#type:o_0# gather_result = {"
            + ", ".join(f"real_gather_src_{i}" for i in range(len(self.in_ports)))
            + ", #end_flag_val#"
            + r"};",
            r"#write_notrans:o_0,gather_result#",
        ]
        return self.get_hls_function(code_in_loop)


class ScatterComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        assert isinstance(input_type, (TupleType, ArrayType))
        real_input_type = input_type
        parallel = False
        if isinstance(input_type, ArrayType):
            assert isinstance(input_type.type_, TupleType)
            real_input_type = input_type.type_
            parallel = True
        ports = ["i_0"]
        for i in range(len(real_input_type.types)):
            ports.append(f"o_{i}")
        # output_type = input_type just for assign
        super().__init__(
            input_type,
            input_type,
            ports,
            parallel,
            specific_port_types={
                f"o_{i}": ArrayType(type_) if parallel else type_
                for i, type_ in enumerate(real_input_type.types)
            },
        )

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = []
        code_in_loop.append(r"#type:i_0# scatter_src = #read:i_0#;")
        code_in_loop.append(r"bool #end_flag_val# = scatter_src.end_flag;")
        for i in range(len(self.out_ports)):
            code_in_loop.append(f"#write:o_{i},scatter_src.ele_{i}#")
        return self.get_hls_function(code_in_loop)


class BinOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    AND = "&"
    OR = "|"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    SL = "<<"
    SR = ">>"
    MIN = "min"
    MAX = "max"

    def __repr__(self) -> str:
        return self.value

    def gen_repr(self, part_a, part_b):
        if self.value in ["min", "max"]:
            translate_dict = {"min": "<", "max": ">"}
            repr = f"(({part_a}) {translate_dict[self.value]} ({part_b}) ? {part_a} : {part_b})"
        else:
            repr = f"{part_a} {self.value} {part_b}"
        return "{ " + repr + " }"

    def gen_repr_forbkd(self, part_a, part_b):
        if self.value in ["min", "max"]:
            translate_dict = {"min": "<", "max": ">"}
            repr = f"(({part_a}) {translate_dict[self.value]} ({part_b}) ? {part_a} : {part_b})"
        else:
            repr = f"{part_a} {self.value} {part_b}"
        return "(" + repr + ")"

    def output_type(self, input_type: DfirType) -> DfirType:
        if self in [
            BinOp.ADD,
            BinOp.SUB,
            BinOp.MUL,
            BinOp.DIV,
            BinOp.AND,
            BinOp.OR,
            BinOp.MIN,
            BinOp.MAX,
            BinOp.SL,
            BinOp.SR,
        ]:
            return input_type
        elif self in [BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.GT, BinOp.LE, BinOp.GE]:
            return BoolType()
        else:
            raise ValueError(f"Unsupported binary operation: {self}")


class BinOpComponent(Component):
    def __init__(self, op: BinOp, input_type: DfirType) -> None:
        if isinstance(input_type, ArrayType):
            parallel = True
            output_type = ArrayType(op.output_type(input_type.type_))
        else:
            parallel = False
            output_type = op.output_type(input_type)
        assert not isinstance(output_type, OptionalType) and not isinstance(output_type, TupleType)
        super().__init__(input_type, output_type, ["i_0", "i_1", "o_0"], parallel)
        self.op = op

    def additional_info(self) -> str:
        return f"op: {self.op}"

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = [
            r"#type:i_0# binop_src_0 = #read:i_0#;",
            r"#type:i_1# binop_src_1 = #read:i_1#;",
            r"bool #end_flag_val# = binop_src_0.end_flag | binop_src_1.end_flag;",
            f"#type_inner:o_0# binop_out = {self.op.gen_repr('binop_src_0#may_ele:i_0#', 'binop_src_1#may_ele:i_1#')};",
            r"#write:o_0,binop_out#",
        ]
        return self.get_hls_function(code_in_loop)


class UnaryOp(Enum):
    NOT = "!"
    NEG = "-"
    CAST_BOOL = "to_bool"
    CAST_INT = "to_int"
    CAST_FLOAT = "to_float"
    SELECT = "select"
    GET_LENGTH = "length"
    GET_ATTR = "get_attr"

    def __repr__(self) -> str:
        return self.value

    def output_type(self, input_type: DfirType) -> DfirType:
        input_available_dict = {
            UnaryOp.NOT: BoolType,
            UnaryOp.NEG: [IntType, FloatType],
            UnaryOp.CAST_BOOL: [IntType, FloatType, BoolType],
            UnaryOp.CAST_INT: [IntType, FloatType, BoolType],
            UnaryOp.CAST_FLOAT: [IntType, FloatType, BoolType],
            UnaryOp.SELECT: TupleType,
            UnaryOp.GET_LENGTH: ArrayType,
            UnaryOp.GET_ATTR: SpecialType,
        }
        assert isinstance(
            input_type, input_available_dict[self]
        ), f"{self}: input type {input_type} should be one of {input_available_dict[self]}"
        if self in [UnaryOp.NOT, UnaryOp.NEG]:
            return input_type
        elif self == UnaryOp.CAST_BOOL:
            return BoolType()
        elif self == UnaryOp.CAST_INT:
            return IntType()
        elif self == UnaryOp.CAST_FLOAT:
            return FloatType()
        elif self == UnaryOp.GET_LENGTH:
            return IntType()
        elif self == UnaryOp.GET_ATTR:
            # get node id from node or get src/dst node id from edge
            return IntType() if input_type.type_name == "node" else SpecialType("node")
        elif self == UnaryOp.SELECT:
            raise RuntimeError("Output type for select operation should be decided outside of UnaryOp")
        else:
            raise ValueError(f"Unsupported unary operation: {self}")


class UnaryOpComponent(Component):
    def __init__(
        self,
        op: UnaryOp,
        input_type: DfirType,
        select_index: Optional[int] = None,
        attr_type: Optional[DfirType] = None,
    ) -> None:
        if isinstance(input_type, ArrayType):
            parallel = True
            real_input_type = input_type.type_
        else:
            parallel = False
            real_input_type = input_type
        if op == UnaryOp.SELECT:
            assert isinstance(real_input_type, TupleType)
            assert select_index is not None
            inside_output_type = real_input_type.types[select_index]
        elif op == UnaryOp.GET_ATTR:
            assert attr_type is not None
            assert select_index is not None
            inside_output_type = attr_type
        else:
            assert not isinstance(
                real_input_type, TupleType
            ), f"input type {real_input_type} is a tuple type for {op}"
            inside_output_type = op.output_type(real_input_type)
        output_type = ArrayType(inside_output_type) if parallel else inside_output_type
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel)
        self.op = op
        self.select_index = select_index

    def additional_info(self) -> str:
        if self.op == UnaryOp.SELECT:
            return [f"op: {self.op}", f"select_index: {self.select_index}"]
        else:
            return f"op: {self.op}"

    def to_hls(self) -> hls.HLSFunction:
        if self.op == UnaryOp.GET_LENGTH:
            code_before_loop = [
                r"uint32_t length = 0;",
            ]
            code_in_loop = [
                r"#read:i_0#;",
                r"length++;",
            ]
            code_after_loop = [
                # r"#output_length# = 1;",
                r"#write:o_0,length#",
            ]
            return self.get_hls_function(code_in_loop, code_before_loop, code_after_loop)
        else:
            trans_dict = {
                UnaryOp.NOT: "{!unary_src#may_ele:i_0#}",
                UnaryOp.NEG: "{-unary_src#may_ele:i_0#}",
                UnaryOp.CAST_BOOL: "{(bool)(unary_src#may_ele:i_0#)}",
                UnaryOp.CAST_INT: "{(int32_t)(unary_src#may_ele:i_0#)}",
                UnaryOp.CAST_FLOAT: "{(ap_fixed<32, 16>)(unary_src#may_ele:i_0#)}",
                UnaryOp.SELECT: f"unary_src.ele_{self.select_index}",
                UnaryOp.GET_ATTR: f"unary_src.{self.select_index}",
            }
            code_in_loop = [
                f"#type:i_0# unary_src = #read:i_0#;",
                f"bool #end_flag_val# = unary_src.end_flag;",
                f"#type_inner:o_0# unary_out = {trans_dict[self.op]};",
                f"#write:o_0,unary_out#",
            ]
            return self.get_hls_function(code_in_loop)


class ConditionalComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        if isinstance(input_type, ArrayType):
            parallel = True
            output_type = ArrayType(OptionalType(input_type.type_))
            cond_type = ArrayType(BoolType())
        else:
            parallel = False
            output_type = OptionalType(input_type)
            cond_type = BoolType()
        super().__init__(
            input_type,
            output_type,
            ["i_data", "i_cond", "o_0"],
            parallel,
            {"i_cond": cond_type},
        )

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = [
            r"#type:i_data# cond_data = #read:i_data#;",
            r"#type:i_cond# cond = #read:i_cond#;",
            r"bool #end_flag_val# = cond_data.end_flag | cond.end_flag;",
            r"#peel:i_data,cond_data,real_cond_data#",
            r"#peel:i_cond,cond,real_cond#",
            r"#opt_type:o_0# cond_result = {real_cond_data, real_cond};",
            r"#write:o_0,cond_result#",
        ]
        return self.get_hls_function(code_in_loop)


class CollectComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        assert isinstance(input_type, ArrayType)
        assert isinstance(input_type.type_, OptionalType)
        output_type = ArrayType(input_type.type_.type_)
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel=True)

    def to_hls(self) -> hls.HLSFunction:
        # code_before_loop = [
        #     r"#output_length# = 0;",
        # ]
        code_in_loop = [
            r"#type:i_0# collect_src = #read:i_0#;",
            r"bool #end_flag_val# = collect_src.end_flag;",
            r"if (collect_src.valid.ele || #end_flag_val#) {",
            r"    #write:o_0,collect_src.data#",
            r"}",
        ]
        return self.get_hls_function(code_in_loop)


class ReduceComponent(Component):
    def __init__(
        self,
        input_type: DfirType,
        accumulated_type: DfirType,
        reduce_key_out_type: DfirType,
    ) -> None:
        assert isinstance(input_type, ArrayType)
        real_input_type = input_type.type_
        super().__init__(
            input_type,
            ArrayType(accumulated_type),
            [
                "i_0",
                "o_0",
                "i_reduce_key_out",
                "i_reduce_transform_out",
                "i_reduce_unit_end",
                "o_reduce_key_in",
                "o_reduce_transform_in",
                "o_reduce_unit_start_0",
                "o_reduce_unit_start_1",
            ],
            parallel=False,
            specific_port_types={
                "i_reduce_key_out": reduce_key_out_type,
                "i_reduce_transform_out": accumulated_type,
                "i_reduce_unit_end": accumulated_type,
                "o_reduce_key_in": real_input_type,
                "o_reduce_transform_in": real_input_type,
                "o_reduce_unit_start_0": accumulated_type,
                "o_reduce_unit_start_1": accumulated_type,
            },
        )

    def to_hls_list(
        self,
        func_key_name: str,
        func_transform_name: str,
        func_unit_name: str,
    ) -> List[hls.HLSFunction]:
        # Generate 1st func for key & transform pre-process
        code_in_loop = [
            r"#type:i_0# reduce_src = #read:i_0#;",
            r"bool #end_flag_val# = reduce_src.end_flag;",
            # f'hls::stream<#type:i_0#> reduce_key_in_stream("reduce_key_in_stream");',
            # r"#pragma HLS STREAM variable=reduce_key_in_stream depth=4",
            # f'hls::stream<#type:i_0#> reduce_transform_in_stream("reduce_transform_in_stream");',
            # r"#pragma HLS STREAM variable=reduce_transform_in_stream depth=4",
            # r"reduce_src.end_flag = true;",
            # f"reduce_key_in_stream.write(reduce_src);",
            # f"reduce_transform_in_stream.write(reduce_src);",
            # f'hls::stream<#type:i_reduce_key_out#> reduce_key_out_stream("reduce_key_out_stream");',
            # r"#pragma HLS STREAM variable=reduce_key_out_stream depth=4",
            # f'hls::stream<#type:i_reduce_transform_out#> reduce_transform_out_stream("reduce_transform_out_stream");',
            # r"#pragma HLS STREAM variable=reduce_transform_out_stream depth=4",
            f"#type:i_reduce_key_out# reduce_key_out;",
            f"#type:i_reduce_transform_out# reduce_transform_out;",
            f"#call_once:{func_key_name},reduce_src,reduce_key_out#;",
            f"#call_once:{func_transform_name},reduce_src,reduce_transform_out#;",
            # r"#type:i_reduce_key_out# reduce_key_out = reduce_key_out_stream.read();",
            r"reduce_key_out.end_flag = #end_flag_val#;",
            r"intermediate_key.write(reduce_key_out);",
            # r"#type:i_reduce_transform_out# reduce_transform_out = reduce_transform_out_stream.read();",
            r"reduce_transform_out.end_flag = #end_flag_val#;",
            r"intermediate_transform.write(reduce_transform_out);",
        ]
        stage_1_func = self.get_hls_function(code_in_loop, name_tail="pre_process")

        # Generate 2nd func for unit-reduce
        code_before_loop = [
            r"#reduce_key_struct# key_mem[MAX_NUM];",
            # r"#pragma HLS ARRAY_PARTITION variable=key_mem dim=0 complete",
            r"#pragma HLS BIND_STORAGE variable = key_mem type = RAM_2P impl = URAM",
            r"#pragma HLS dependence variable=key_mem inter false",
            r"#reduce_key_struct# key_buffer[L + 1];",
            r"#pragma HLS ARRAY_PARTITION variable=key_buffer dim=0 complete",
            r"uint32_t i_buffer[L + 1];",
            r"#pragma HLS ARRAY_PARTITION variable=i_buffer dim=0 complete",
            r"for (int i_clear_buffer = 0; i_clear_buffer < L + 1; i_clear_buffer++) {",
            r"#pragma HLS UNROLL",
            r"    i_buffer[i_clear_buffer] = MAX_NUM + 1;",
            r"}",
            # r"#pragma HLS ARRAY_PARTITION variable=key_mem complete dim=0",
            r"CLEAR_REDUCE_VALID: for (int i_reduce_clear = 0; i_reduce_clear < MAX_NUM; i_reduce_clear++) {",
            r"#pragma HLS UNROLL",
            r"    key_mem[i_reduce_clear].valid.ele = 0;",
            r"}",
        ]
        code_in_loop = [
            # the reduce_key_struct is {key, valid}, the loop uses one loop ahead
            # to clear the valid bit to 0 with pipeline
            f"#type:i_reduce_key_out# reduce_key_out = #read:intermediate_key#;",
            f"#type:i_reduce_transform_out# reduce_transform_out = #read:intermediate_transform#;",
            r"bool #end_flag_val# = reduce_key_out.end_flag | reduce_transform_out.end_flag;",
            # r"bool merged = false;",
            r"#peel:i_reduce_key_out,reduce_key_out,real_reduce_key_out#",
            r"#peel:i_reduce_transform_out,reduce_transform_out,real_reduce_transform_out#",
            # r"SCAN_BRAM_INTER_LOOP: for (int i_in_reduce = 0; i_in_reduce < MAX_NUM; i_in_reduce++) {",
            # r"#pragma HLS PIPELINE",
            # r"    #reduce_key_struct# cur_ele = key_mem[i_in_reduce];",
            # r"    if (!merged && !cur_ele.valid.ele) {",
            # r"        key_mem[i_in_reduce].valid.ele = 1;",
            # r"        key_mem[i_in_reduce].key#may_ele:i_reduce_key_out# = real_reduce_key_out;",
            # r"        key_mem[i_in_reduce].data#may_ele:i_reduce_transform_out# = real_reduce_transform_out;",
            # r"        merged = true;",
            # r"    } else if (!merged && cur_ele.valid.ele && #cmpeq:i_reduce_key_out,cur_ele.key,real_reduce_key_out#) {",
            # # new a stream to call the reduce unit
            # '        hls::stream<#type:o_reduce_unit_start_0#> reduce_unit_stream_0("reduce_unit_stream_0");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_0 depth=4",
            # '        hls::stream<#type:o_reduce_unit_start_1#> reduce_unit_stream_1("reduce_unit_stream_1");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_1 depth=4",
            # '        hls::stream<#type:i_reduce_unit_end#> reduce_unit_stream_out("reduce_unit_stream_out");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_out depth=4",
            # r"        #write:reduce_unit_stream_0,cur_ele.data#may_ele:i_reduce_transform_out#,#type:i_reduce_transform_out##",
            # r"        #write:reduce_unit_stream_1,real_reduce_transform_out,#type:i_reduce_transform_out##",
            # f"        #call_once:{func_unit_name},reduce_unit_stream_0,reduce_unit_stream_1,reduce_unit_stream_out#;",
            # r"        #type:i_reduce_unit_end# reduce_unit_out = #read:reduce_unit_stream_out#;",
            # r"        #peel:i_reduce_unit_end,reduce_unit_out,real_reduce_unit_out#",
            # r"        key_mem[i_in_reduce].data#may_ele:i_reduce_transform_out# = real_reduce_unit_out;",
            # r"        merged = true;",
            # r"    }",
            r"#reduce_key_struct# old_ele = key_mem[real_reduce_key_out#may_ele:i_reduce_key_out#];",
            r"for (int i_search_buffer = 0; i_search_buffer < L + 1; i_search_buffer++) {",
            r"#pragma HLS UNROLL",
            r"    {",
            r"        if (real_reduce_key_out#may_ele:i_reduce_key_out# == i_buffer[i_search_buffer]) old_ele = key_buffer[i_search_buffer];",
            r"    }",
            r"}",
            r"for (int i_move_buffer = 0; i_move_buffer < L; i_move_buffer++) {",
            r"#pragma HLS UNROLL",
            r"    {",
            r"        i_buffer[i_move_buffer] = i_buffer[i_move_buffer + 1];",
            r"        key_buffer[i_move_buffer] = key_buffer[i_move_buffer + 1];",
            r"    }",
            r"}",
            r"#reduce_key_struct# new_ele;",
            r"if (!old_ele.valid.ele) {",
            r"    new_ele.valid.ele = 1;",
            r"    new_ele.data = real_reduce_transform_out;",
            r"} else {",
            # '    hls::stream<#type:o_reduce_unit_start_0#> reduce_unit_stream_0("reduce_unit_stream_0");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_0 depth=1",
            # '    hls::stream<#type:o_reduce_unit_start_1#> reduce_unit_stream_1("reduce_unit_stream_1");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_1 depth=1",
            # '    hls::stream<#type:i_reduce_unit_end#> reduce_unit_stream_out("reduce_unit_stream_out");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_out depth=1",
            # r"    #write:reduce_unit_stream_0,old_ele.data,#type:i_reduce_transform_out##",
            # r"    #write:reduce_unit_stream_1,real_reduce_transform_out,#type:i_reduce_transform_out##",
            r"    #write_nostream:old_ele_data,old_ele.data,#type:i_reduce_transform_out##",
            r"    #write_nostream:new_ele_data,real_reduce_transform_out,#type:i_reduce_transform_out##",
            r"    #type:i_reduce_unit_end# reduce_unit_out;",
            f"    #call_once:{func_unit_name},old_ele_data,new_ele_data,reduce_unit_out#;",
            # r"    #type:i_reduce_unit_end# reduce_unit_out = #read:reduce_unit_stream_out#;",
            r"    #peel:i_reduce_unit_end,reduce_unit_out,real_reduce_unit_out#",
            r"    new_ele.data = real_reduce_unit_out;",
            r"}",
            r"key_mem[real_reduce_key_out#may_ele:i_reduce_key_out#] = new_ele;",
            r"key_buffer[L] = new_ele;",
            r"i_buffer[L] = real_reduce_key_out#may_ele:i_reduce_key_out#;",
        ]
        code_after_loop = [
            # r"#output_length# = 0;",
            r"WRITE_KEY_MEM_LOOP: for (int i_write_key_mem = 0; i_write_key_mem < MAX_NUM; i_write_key_mem++) {",
            r"#pragma HLS PIPELINE",
            r"    if (key_mem[i_write_key_mem].valid.ele) {",
            r"        #write_noend:o_0,key_mem[i_write_key_mem].data#may_ele:o_0##",
            # r"        #output_length#++;",
            r"    }",
            r"}",
            r"#write:o_0,key_mem[0].data#may_ele:o_0##",
        ]
        stage_2_func = self.get_hls_function(
            code_in_loop, code_before_loop, code_after_loop, name_tail="unit_reduce"
        )
        return [stage_1_func, stage_2_func]


class PlaceholderComponent(Component):
    def __init__(self, data_type: DfirType) -> None:
        super().__init__(data_type, data_type, ["i_0", "o_0"])

    def to_hls(self) -> hls.HLSFunction:
        assert False, "PlaceholderComponent should not be used in HLS"


class UnusedEndMarkerComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, None, ["i_0"])

    def to_hls(self) -> hls.HLSFunction:
        assert False, "UnusedEndMarkerComponent should not be used in HLS"

```

`CCFSys2025_GraphyFlow/graphyflow/dataflow_ir_datatype.py`:

```py
from __future__ import annotations
from typing import List


class DfirType:
    def __init__(self, type_name, is_optional: bool = False, is_basic_type=True) -> None:
        self.type_name = type_name
        self.is_optional = is_optional
        self.is_basic_type = is_basic_type

    def __eq__(self, other: DfirType) -> bool:
        return self.type_name == other.type_name

    def __hash__(self) -> int:
        return hash(self.type_name)

    def __repr__(self) -> str:
        return self.type_name


class SpecialType(DfirType):
    """Node or Edge"""

    def __init__(self, type_name: str) -> None:
        assert type_name in ["node", "edge"]
        super().__init__(type_name, is_basic_type=False)


class IntType(DfirType):
    def __init__(self) -> None:
        super().__init__("Int")


class FloatType(DfirType):
    def __init__(self) -> None:
        super().__init__("Float")


class BoolType(DfirType):
    def __init__(self) -> None:
        super().__init__("Bool")


class OptionalType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        assert not isinstance(type_, OptionalType)
        super().__init__(f"Optional<{type_.type_name}>", is_optional=True, is_basic_type=False)
        self.type_ = type_


class TensorType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        assert not isinstance(type_, TensorType)
        super().__init__(f"Tensor<{type_.type_name}>", is_basic_type=False)
        self.type_ = type_


class TupleType(DfirType):
    def __init__(self, types: List[DfirType]) -> None:
        super().__init__(f"Tuple<{', '.join([t.type_name for t in types])}>", is_basic_type=False)
        self.types = types


class ArrayType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        super().__init__(f"Array<{type_.type_name}>", is_basic_type=False)
        self.type_ = type_

```

`CCFSys2025_GraphyFlow/graphyflow/global_graph.py`:

```py
from __future__ import annotations
from typing import Callable, List, Optional, Dict
from uuid import UUID
import uuid as uuid_lib
from graphyflow.graph_types import *
from graphyflow.lambda_func import parse_lambda, Tracer, format_lambda, lambda_to_dfir
import graphyflow.dataflow_ir as dfir


class Node:
    """每个计算图的节点可以单或多输入，但是只有一样输出。任何将该节点作为pred_nodes的节点都将接受到该节点的输出的一份复制"""

    def __init__(
        self,
    ) -> None:
        self.uuid = uuid_lib.uuid4()
        self.is_simple = False
        self._pred_nodes = []
        self._lambda_funcs = []

    def set_pred_nodes(self, pred_nodes: List[Node]) -> None:
        self._pred_nodes = pred_nodes

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    @property
    def preds(self) -> List[Node]:
        return self._pred_nodes

    @property
    def lambdas(self) -> List[Dict]:
        return self._lambda_funcs

    def __repr__(self) -> str:
        formatted_lambdas = "\n".join(format_lambda(lambda_func) for lambda_func in self._lambda_funcs)
        return f"Node(name={self.class_name}, preds={[node.class_name for node in self._pred_nodes]}, lambdas = {formatted_lambdas})"


class Inputer(Node):
    def __init__(self, input_type: Union[BasicNode, BasicEdge]):
        self.input_type = input_type
        super().__init__()

    def to_dfir(
        self,
        placeholder: List[dfir.DfirType],
        props: Tuple[Dict[str, Any], Dict[str, Any]],
    ) -> dfir.DfirNode:
        if isinstance(self.input_type, BasicNode):
            input_type = dfir.SpecialType("node")
        elif isinstance(self.input_type, BasicEdge):
            input_type = dfir.SpecialType("edge")
        else:
            raise RuntimeError("Input type must be BasicNode or BasicEdge")
        input_type = dfir.ArrayType(input_type)
        io_comp = dfir.IOComponent(dfir.IOComponent.IOType.INPUT, input_type)
        return dfir.ComponentCollection([io_comp], [], [io_comp.ports[0]])


class Updater(Node):
    def __init__(self, type_: str, attrs: List[str]):
        assert type_ in ["node", "edge"]
        self.type_ = type_
        self.attrs = attrs
        super().__init__()


class GetLength(Node):
    def __init__(self) -> None:
        super().__init__()
        self.is_simple = True

    def to_dfir(self, input_type: dfir.DfirType) -> dfir.DfirNode:
        u_comp = dfir.UnaryOpComponent(dfir.UnaryOp.GET_LENGTH, input_type)
        return dfir.ComponentCollection([u_comp], [u_comp.in_ports[0]], [u_comp.out_ports[0]])


class Filter(Node):
    def __init__(self, filter_func: Callable[[List[DataElement]], DataElement]):
        self.filter_func = parse_lambda(filter_func)
        super().__init__()
        self._lambda_funcs.append(self.filter_func)

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        assert isinstance(input_type, dfir.ArrayType)
        assert len(self._lambda_funcs) == 1
        if len(self._lambda_funcs[0]["input_ids"]) == 1:
            dfirs = lambda_to_dfir(self._lambda_funcs[0], [input_type], props[0], props[1])
            assert len(dfirs.outputs) == 1
            assert dfirs.outputs[0].data_type == dfir.ArrayType(dfir.BoolType())

            copy_comp = dfir.CopyComponent(input_type)
            cond_comp = dfir.ConditionalComponent(input_type)
            collect_comp = dfir.CollectComponent(cond_comp.output_type)

            assert len(dfirs.inputs) == 1
            copy_ports = {"o_0": dfirs.inputs[0]}
            dfirs.add_front(copy_comp, copy_ports)

            cond_ports = {"i_data": copy_comp.ports[2], "i_cond": dfirs.outputs[0]}
            dfirs.add_back(cond_comp, cond_ports)

            collect_ports = {"i_0": cond_comp.ports[2]}
            dfirs.add_back(collect_comp, collect_ports)

            return dfirs
        else:
            scatter_comp = dfir.ScatterComponent(input_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            dfirs = lambda_to_dfir(self._lambda_funcs[0], scatter_out_types, props[0], props[1])
            assert len(dfirs.outputs) == 1
            assert dfirs.outputs[0].data_type == dfir.ArrayType(dfir.BoolType())

            copy_comp = dfir.CopyComponent(input_type)
            cond_comp = dfir.ConditionalComponent(input_type)
            collect_comp = dfir.CollectComponent(cond_comp.output_type)

            dfirs.add_front(
                scatter_comp,
                {f"o_{i}": dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )

            copy_ports = {"o_0": scatter_comp.ports[0]}
            dfirs.add_front(copy_comp, copy_ports)

            cond_ports = {"i_data": copy_comp.ports[2], "i_cond": dfirs.outputs[0]}
            dfirs.add_back(cond_comp, cond_ports)

            collect_ports = {"i_0": cond_comp.ports[2]}
            dfirs.add_back(collect_comp, collect_ports)

            return dfirs


class Map_(Node):
    def __init__(self, map_func: Callable[[List[DataElement]], DataElement]):
        self.map_func = parse_lambda(map_func)
        super().__init__()
        self.lambdas.append(self.map_func)

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        assert isinstance(input_type, dfir.ArrayType), f"{input_type} is not an array type"
        assert len(self._lambda_funcs) == 1
        if len(self._lambda_funcs[0]["input_ids"]) == 1:
            dfirs = lambda_to_dfir(self._lambda_funcs[0], [input_type], props[0], props[1])
        else:
            scatter_comp = dfir.ScatterComponent(input_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            dfirs = lambda_to_dfir(self._lambda_funcs[0], scatter_out_types, props[0], props[1])
            dfirs.add_front(
                scatter_comp,
                {f"o_{i}": dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )
        if len(dfirs.outputs) > 1:
            gather_comp = dfir.GatherComponent(dfirs.output_types)
            dfirs.add_back(
                gather_comp,
                {f"i_{i}": dfirs.outputs[i] for i in range(len(dfirs.outputs))},
            )
        return dfirs


class ReduceBy(Node):
    """
    reduce_key: x -> y, determine according to which element (is same) to group by and reduce, if want to reduce all, give lambda x: 1
    reduce_transform: x -> y, determine which element should be reduced
    reduce_method: a, b -> c, reduce two elements, should be commutative and associative
    """

    def __init__(
        self,
        reduce_key: Callable[[List[DataElement]], DataElement],
        reduce_transform: Callable[[List[DataElement]], DataElement],
        reduce_method: Callable[[List[DataElement]], DataElement],
    ):
        self.reduce_key = parse_lambda(reduce_key)
        self.reduce_transform = parse_lambda(reduce_transform)
        self.reduce_method = parse_lambda(reduce_method)
        super().__init__()
        self.lambdas.extend([self.reduce_key, self.reduce_transform, self.reduce_method])

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        assert isinstance(input_type, dfir.ArrayType)
        assert len(self._lambda_funcs) == 3

        element_type = input_type.type_
        if len(self.reduce_key["input_ids"]) == 1:
            reduce_key_dfirs = lambda_to_dfir(self.reduce_key, [element_type], props[0], props[1])
        else:
            scatter_comp = dfir.ScatterComponent(element_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            reduce_key_dfirs = lambda_to_dfir(self.reduce_key, scatter_out_types, props[0], props[1])
            reduce_key_dfirs.add_front(
                scatter_comp,
                {f"o_{i}": reduce_key_dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )
        assert len(reduce_key_dfirs.inputs) == 1
        assert len(reduce_key_dfirs.outputs) == 1
        reduce_key_in_port = reduce_key_dfirs.inputs[0]
        reduce_key_out_port = reduce_key_dfirs.outputs[0]
        reduce_key_out_type = reduce_key_dfirs.output_types[0]

        if len(self.reduce_transform["input_ids"]) == 1:
            reduce_transform_dfirs = lambda_to_dfir(
                self.reduce_transform,
                [element_type],
                props[0],
                props[1],
                scatter_outputs=False,
            )
        else:
            scatter_comp = dfir.ScatterComponent(element_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            reduce_transform_dfirs = lambda_to_dfir(
                self.reduce_transform,
                scatter_out_types,
                props[0],
                props[1],
                scatter_outputs=False,
            )
            reduce_transform_dfirs.add_front(
                scatter_comp,
                {f"o_{i}": reduce_transform_dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )
        assert len(reduce_transform_dfirs.inputs) == 1
        assert len(reduce_transform_dfirs.outputs) == 1
        transform_input_port = reduce_transform_dfirs.inputs[0]
        transform_output_port = reduce_transform_dfirs.outputs[0]
        transform_type = transform_output_port.data_type

        assert len(self.reduce_method["input_ids"]) == 2
        reduce_method_dfirs = lambda_to_dfir(
            self.reduce_method,
            [transform_type, transform_type],
            props[0],
            props[1],
            scatter_outputs=False,
        )
        assert len(reduce_method_dfirs.inputs) == 2
        assert len(reduce_method_dfirs.outputs) == 1
        accumulate_input_port_0 = reduce_method_dfirs.inputs[0]
        accumulate_input_port_1 = reduce_method_dfirs.inputs[1]
        accumulate_output_port = reduce_method_dfirs.outputs[0]

        dfirs = reduce_transform_dfirs
        dfirs = dfirs.concat(reduce_method_dfirs, [])
        dfirs = dfirs.concat(reduce_key_dfirs, [])
        reduce_comp = dfir.ReduceComponent(input_type, transform_type, reduce_key_out_type)
        dfirs.add_front(
            reduce_comp,
            {
                "o_reduce_key_in": reduce_key_in_port,
                "o_reduce_transform_in": transform_input_port,
                "o_reduce_unit_start_0": accumulate_input_port_0,
                "o_reduce_unit_start_1": accumulate_input_port_1,
            },
        )
        dfirs.add_back(
            reduce_comp,
            {
                "i_reduce_key_out": reduce_key_out_port,
                "i_reduce_transform_out": transform_output_port,
                "i_reduce_unit_end": accumulate_output_port,
            },
        )
        return dfirs


class Merge(Node):
    def __init__(self):
        super().__init__()

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        pass


class Append(Node):
    def __init__(self):
        super().__init__()

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        pass


class GlobalGraph:
    def __init__(self, properties: Optional[Dict[str, Dict[str, Any]]] = None):
        self.input_nodes = []  # by UUID
        self.nodes = {}  # Each node represents a method, nodes = {uuid: node}
        self.node_properties = {}
        self.edge_properties = {}
        self.added_input = False
        self.input_names = ["edge", "node"]
        if properties:
            self.handle_properties(properties)

    def handle_properties(self, properties: Dict[str, Dict[str, Any]]):
        assert not self.added_input, "Properties must be set before adding any input"
        if "edge" not in properties.keys():
            properties["edge"] = {}
        if "node" not in properties.keys():
            properties["node"] = {}
        properties["edge"]["src"] = dfir.SpecialType("node")
        properties["edge"]["dst"] = dfir.SpecialType("node")
        properties["node"]["id"] = dfir.IntType()
        for prop_name, prop_info in properties.items():
            assert prop_name in ["node", "edge"]
            if prop_name == "node":
                self.node_properties = prop_info
            else:
                self.edge_properties = prop_info

    def pseudo_element(self, **kwargs) -> PseudoElement:
        return PseudoElement(graph=self, **kwargs)

    def add_graph_input(self, type_: str, **kwargs) -> PseudoElement:
        assert type_ in ["edge", "node"]
        self.added_input = True
        property_infos = self.node_properties if type_ == "node" else self.edge_properties
        data_types = [BasicData(prop_type) for _, prop_type in property_infos.items()]
        return self.pseudo_element(
            cur_node=Inputer(input_type=(BasicNode(data_types) if type_ == "node" else BasicEdge(data_types)))
        )

    def add_input(self, name: str, type_: str):
        assert name not in self.input_names
        self.input_names.append(name)
        self.input_types[name] = type_
        return self.pseudo_element(cur_node=Inputer(input_type=type_))

    def assign_node(self, node: Node):
        assert node.uuid not in self.nodes
        self.nodes[node.uuid] = node

    def finish_init(self, datas: PseudoElement, attrs: Dict[str, List[str]]):
        for attr_name, attr_types in attrs.items():
            datas._assign_successor(Updater(attr_name, attr_types))

    def finish_iter(
        self,
        datas: PseudoElement,
        attrs: Dict[str, List[str]],
        end_marker: PseudoElement,
    ):
        for attr_name, attr_types in attrs.items():
            datas._assign_successor(Updater(attr_name, attr_types))

    def topo_sort_nodes(self) -> List[Node]:
        result = []
        waitings = list(self.nodes.items())
        while waitings:
            new_ones = []
            for nid, n in waitings:
                if all((pred.uuid, pred) in result for pred in n.preds):
                    new_ones.append((nid, n))
            waitings = [w for w in waitings if w not in new_ones]
            result.extend(new_ones)
        return result

    def to_dfir(self) -> dfir.ComponentCollection:
        nodes = self.topo_sort_nodes()
        node_dfirs = {}
        added_nodes = []
        for nid, n in nodes:
            input_types = []
            if len(n.preds) > 0:
                assert len(n.preds) == 1
                input_types.extend(node_dfirs[n.preds[0].uuid].output_types)
                assert len(input_types) == 1, f"{n.class_name} {input_types}"
                input_types = input_types[0]
            node_dfirs[nid] = n.to_dfir(input_types, (self.node_properties, self.edge_properties))
            if len(n.preds) > 0:
                node_dfirs[nid] = node_dfirs[n.preds[0].uuid].concat(
                    node_dfirs[nid],
                    [
                        (
                            node_dfirs[n.preds[0].uuid].outputs[0],
                            node_dfirs[nid].inputs[0],
                        )
                    ],
                )
                added_nodes.append(n.preds[0].uuid)
        result_dfirs = [d for nid, d in node_dfirs.items() if nid not in added_nodes]
        return result_dfirs

    def __repr__(self) -> str:
        return f"GlobalGraph(nodes={self.nodes})"


class PseudoElement:
    def __init__(
        self,
        graph: GlobalGraph,
        cur_node: Optional[Node] = None,
    ):
        self.graph = graph
        self.cur_node = cur_node
        if cur_node:
            self.graph.assign_node(cur_node)

    def __repr__(self) -> str:
        return f"PseudoElement(cur_node={self.cur_node.class_name})"

    def _assign_cur_node(self, cur_node: Node):
        assert self.cur_node is None
        self.graph.assign_node(cur_node)
        self.cur_node = cur_node

    def _assign_successor(self, succ_node: Node):
        if self.cur_node:
            succ_node.set_pred_nodes([self.cur_node])
        return self.graph.pseudo_element(cur_node=succ_node)

    def length(self) -> PseudoElement:
        return self._assign_successor(GetLength())

    def filter(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Filter(**kvargs))

    def map_(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Map_(**kvargs))

    def reduce_by(self, **kvargs) -> PseudoElement:
        return self._assign_successor(ReduceBy(**kvargs))

    def merge(self, other: PseudoElement) -> PseudoElement:
        assert self.cur_node is not None and other.cur_node is not None
        self.cur_node.set_pred_nodes([other.cur_node])
        return self.graph.pseudo_element(cur_node=Merge())

    def append(self, other: PseudoElement) -> PseudoElement:
        assert self.cur_node is not None and other.cur_node is not None
        self.cur_node.set_pred_nodes([other.cur_node])
        return self.graph.pseudo_element(cur_node=Append())

    def to_tracer(self) -> Tracer:
        return Tracer(node_type="pseudo", pseudo_element=self)

```

`CCFSys2025_GraphyFlow/graphyflow/graph_types.py`:

```py
from typing import Any, Tuple, Union, Type
from uuid import UUID
import uuid as uuid_lib
from abc import ABC
import graphyflow.dataflow_ir as dfir


class TransportType(ABC):
    def __init__(self):
        self.data = None


class BasicData(TransportType):
    """最基础的数据单元, 可以是int/float"""

    def __init__(self, data_type: Type[Any]):
        self._data_type = data_type

    @property
    def data_type(self):
        return self._data_type

    def to_dfir(self) -> dfir.DfirType:
        assert self.data_type in [dfir.IntType(), dfir.FloatType(), dfir.BoolType()]
        return self.data_type

    def __repr__(self):
        return f"BasicData({self._data_type})"


class BasicNode(TransportType):
    """图节点单元，代表某个节点，也包括若干个 BasicData"""

    def __init__(self, data_types: Union[BasicData, Tuple[BasicData]] = None):
        if isinstance(data_types, BasicData):
            data_types = [data_types]
        assert all(isinstance(d, BasicData) for d in data_types)
        self.data_types = data_types

    def __repr__(self):
        return f"BasicNode({self.data_types})"


class BasicEdge(TransportType):
    """图边单元，代表某个边，也包括若干个 BasicData"""

    def __init__(self, data_types: Union[BasicData, Tuple[BasicData]] = None):
        if isinstance(data_types, BasicData):
            data_types = [data_types]
        assert all(isinstance(d, BasicData) for d in data_types)
        self.data_types = data_types

    def __repr__(self):
        return f"BasicEdge({self.data_types})"


class BasicArray(TransportType):
    """数组单元, 包括若干个相同类型的 BasicData 或 BasicNode 或 BasicEdge 的集合"""

    def __init__(self, data_type: Union[BasicData, BasicNode, BasicEdge]):
        assert isinstance(data_type[0], (BasicData, BasicNode, BasicEdge))
        assert all([isinstance(d, type(data_type[0])) for d in data_type])
        if isinstance(data_type[0], BasicData):
            assert all(d.data_type == data_type[0].data_type for d in data_type)
        self.data_type = data_type

    def __repr__(self):
        return f"BasicArray({self.data_type})"


class DataElement:
    """每条边和每个节点传输的单位是 DataElement 对象, 内部可能包括若干个 BasicData 或 BasicArray"""

    def __init__(self, element_types: Tuple[Union[Any, Any]] = ()):
        self.element_types = element_types
        assert all([isinstance(e, (BasicData, BasicNode, BasicEdge, BasicArray)) for e in element_types])

    def __repr__(self):
        return f"DataElement({self.element_types})"

```

`CCFSys2025_GraphyFlow/graphyflow/hls_utils.py`:

```py
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


class HLSDataType(Enum):
    UINT = "uint32_t"
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    BOOL = "bool"
    EDGE = "edge_t"
    NODE = "node_t"

    def __repr__(self) -> str:
        return self.value

    def gen_basic_print(hls_name, type_name):
        if type_name == HLSDataType.INT or type_name == HLSDataType.BOOL:
            return f"""#ifdef PRINT_DEFS
#define print_{hls_name}(var) \\
    printf("%d", var.ele);
#endif"""
        elif type_name == HLSDataType.FLOAT:
            return f"""#ifdef PRINT_DEFS
#define print_{hls_name}(var) \\
    printf("%lf", (double)var.ele);
#endif"""
        else:
            assert False, f"Not supported for ({hls_name}, {type_name})"


STD_TYPE_TRANSLATE_MAP = (
    (dftype.IntType(), HLSDataType.INT),
    (dftype.FloatType(), HLSDataType.FLOAT),
    (dftype.BoolType(), HLSDataType.BOOL),
    (dftype.SpecialType("node"), HLSDataType.NODE),
    (dftype.SpecialType("edge"), HLSDataType.EDGE),
)
STD_TYPES = ["uint32_t", "int32_t", "ap_fixed<32, 16>", "bool"]


class HLSDataTypeManager:
    _cnt = 0

    def __init__(
        self,
        node_props: Dict[str, dftype.DfirType],
        edge_props: Dict[str, dftype.DfirType],
    ) -> None:
        self.define_map = {}
        self.translate_map = {}
        self.type_preds = {}
        self.to_outer_type = {}
        self.basic_basic_types = {}
        self.basic_type_names = []
        for dfir_type, hls_type in STD_TYPE_TRANSLATE_MAP:
            type_name = f"basic_{hls_type.value[:5]}_t"
            if not isinstance(dfir_type, dfir.SpecialType):
                self.basic_basic_types[type_name] = hls_type
            self.define_map[dfir_type] = (
                [f"    {hls_type.value} ele;"],
                f"{type_name}",
            )
            self.type_preds[type_name] = []
            self.translate_map[dfir_type] = type_name
            self.basic_type_names.append(type_name)
        self.node_properties = {
            p_name: self.from_dfir_type(p_type, outer=False) for p_name, p_type in node_props.items()
        }
        self.edge_properties = {
            p_name: self.from_dfir_type(p_type, outer=False) for p_name, p_type in edge_props.items()
        }

    @classmethod
    def get_next_id(cls) -> int:
        cls._cnt += 1
        return cls._cnt

    def get_all_defines(self) -> List[str]:
        def gen_define(def_map_ele):
            return "typedef struct {\n" + "\n".join(def_map_ele[0]) + "\n} " + f"{def_map_ele[1]};"

        # first generate node_t & edge_t define, add to type_preds
        self.define_map[dftype.SpecialType("node")] = (
            [f"    {t} {p_name};" for p_name, t in self.node_properties.items()],
            "basic_node__t",
        )
        self.type_preds["basic_node__t"] = list(self.node_properties.values())
        self.type_preds["basic_node__t"] = [t for t in self.type_preds["basic_node__t"] if t not in STD_TYPES]
        self.translate_map[dftype.SpecialType("node")] = "basic_node__t"
        self.define_map[dftype.SpecialType("edge")] = (
            [f"    {t} {p_name};" for p_name, t in self.edge_properties.items()],
            "basic_edge__t",
        )
        self.type_preds["basic_edge__t"] = list(self.edge_properties.values())
        self.type_preds["basic_edge__t"] = [t for t in self.type_preds["basic_edge__t"] if t not in STD_TYPES]
        self.translate_map[dftype.SpecialType("edge")] = "basic_edge__t"
        # then iterate all translate_map
        all_defines = []
        waitings = list(self.translate_map.items())
        finished = self.basic_type_names[:]

        # special deal with print_bool()
        all_defines.append(
            """#ifdef PRINT_DEFS
#define print_bool(var) \\
    printf("%d", var);
#endif"""
        )

        # print(finished)
        def gen_print_func(def_map_ele):
            if def_map_ele[1] in self.basic_basic_types.keys():
                return HLSDataType.gen_basic_print(def_map_ele[1], self.basic_basic_types[def_map_ele[1]])
            params = []
            for one_def_ele in def_map_ele[0]:
                print(f"{one_def_ele=}")
                one_def_ele = one_def_ele[:-1].strip().split(" ")
                assert len(one_def_ele) >= 2, f"{one_def_ele} len != 2"
                params.append(("".join(one_def_ele[:-1]), one_def_ele[-1]))
            LB, RB = "{", "}"
            print(def_map_ele[1])
            print(params)
            return (
                "#ifdef PRINT_DEFS\n"
                + f"#define print_{def_map_ele[1]}(var) \\\n"
                + f'    printf("{def_map_ele[1]} {LB}\\n"); \\\n'
                + ' printf(", "); \\\n'.join(
                    f"    print_{sub_type}(var.{param});" for sub_type, param in params
                )
                + f' \\\n    printf("{RB}\\n"); \n'
                + "#endif"
            )

        while waitings:
            dfir_type, type_name = waitings.pop(0)
            assert (
                dfir_type in self.define_map
            ), f"dfir_type {dfir_type} with type_name {type_name} not in define map"
            # print(dfir_type, type_name, self.type_preds[type_name])
            if not all(t in finished for t in self.type_preds[type_name]) and type_name not in [
                "basic_int32_t",
                "basic_ap_fi_t",
                "basic_bool_t",
            ]:
                waitings.append((dfir_type, type_name))
                continue
            all_defines.append(gen_define(self.define_map[dfir_type]))
            all_defines.append(gen_print_func(self.define_map[dfir_type]))
            finished.append(type_name)
        for dfir_type, outer_name in self.to_outer_type.items():
            ori_type_name = self.translate_map[dfir_type]
            assert ori_type_name in finished
            new_map_ele = list(self.define_map[dfir_type])
            ori_params = []
            for param_line in new_map_ele[0]:
                ori_params.append(param_line.split(" ")[-1][:-1])
            # add transition funcs
            transition_func1 = (
                f"#define {ori_type_name}_to_{outer_name}(origin_name, new_name, end_flag_val) \\\n"
                + f"    {outer_name} new_name;\\\n"
                + "\\\n".join(f"    new_name.{param} = origin_name.{param};" for param in ori_params)
                + "\\\n    new_name.end_flag = end_flag_val;\n\n"
            )
            transition_func2 = (
                f"#define {outer_name}_to_{ori_type_name}(origin_name, new_name) \\\n"
                + f"    {ori_type_name} new_name;\\\n"
                + "\\\n".join(f"    new_name.{param} = origin_name.{param};" for param in ori_params)
            )
            new_map_ele[0].append("    bool end_flag;")
            new_map_ele[1] = outer_name
            all_defines.append(gen_define(new_map_ele))
            all_defines.append(gen_print_func(new_map_ele))
            all_defines.append(transition_func1 + transition_func2)
        return all_defines

    def from_dfir_type(
        self,
        dfir_type: dftype.DfirType,
        outer=True,
        sub_names: Optional[List[str]] = None,
    ) -> str:
        if isinstance(dfir_type, dftype.ArrayType):
            dfir_type = dfir_type.type_
        if dfir_type in self.translate_map:
            if outer:
                if not dfir_type in self.to_outer_type:
                    self.to_outer_type[dfir_type] = "outer_" + self.translate_map[dfir_type]
                return self.to_outer_type[dfir_type]
            return self.translate_map[dfir_type]
        else:
            assert isinstance(
                dfir_type, (dftype.TupleType, dftype.OptionalType)
            ), f"Unsupported type: {dfir_type}"
            assert dfir_type not in self.define_map, f"Type {dfir_type} already defined"
            if isinstance(dfir_type, dftype.TupleType):
                sub_types = [self.from_dfir_type(t, outer=False) for t in dfir_type.types]
                name_id = HLSDataTypeManager.get_next_id()
                type_name = f'tuple_{"".join(st[:1] for st in sub_types)}_{name_id}_t'
                self.translate_map[dfir_type] = type_name
                self.type_preds[type_name] = [t for t in sub_types if t not in STD_TYPES]
                self.define_map[dfir_type] = (
                    ([f"    {t} ele_{i};" for i, t in enumerate(sub_types)], type_name)
                    if sub_names is None
                    else (
                        [f"    {t} {sub_names[i]};" for i, t in enumerate(sub_types)],
                        type_name,
                    )
                )
                if outer:
                    if not dfir_type in self.to_outer_type:
                        self.to_outer_type[dfir_type] = "outer_" + self.translate_map[dfir_type]
                    return self.to_outer_type[dfir_type]
                return type_name
            elif isinstance(dfir_type, dftype.OptionalType):
                sub_type = self.from_dfir_type(dfir_type.type_, False)
                type_name = f"opt__of_{sub_type[:3]}_t"
                self.translate_map[dfir_type] = type_name
                self.type_preds[type_name] = [sub_type]
                self.define_map[dfir_type] = (
                    [f"    {sub_type} data;", "    basic_bool_t valid;"],
                    type_name,
                )
                if outer:
                    if not dfir_type in self.to_outer_type:
                        self.to_outer_type[dfir_type] = "outer_" + self.translate_map[dfir_type]
                    return self.to_outer_type[dfir_type]
                return type_name
            else:
                raise ValueError(f"Unsupported type: {dfir_type}")


class HLSFunction:
    def __init__(
        self,
        name: str,
        comp: dfir.Component,
        code_in_loop: List[str],
        code_before_loop: Optional[List[str]] = [],
        code_after_loop: Optional[List[str]] = [],
    ) -> None:
        assert name not in global_hls_config.functions, f"Function {name} already exists"
        self.name = name
        self.code_in_loop = code_in_loop
        self.code_before_loop = code_before_loop
        self.code_after_loop = code_after_loop
        self.comp = comp
        self.params = []
        # self.change_length = any(
        #     "#output_length#" in line
        #     for line in (code_before_loop + code_in_loop + code_after_loop)
        # )
        global_hls_config.functions[name] = self

    def __repr__(self) -> str:
        return f"HLSFunction(name={self.name}, inputs={self.inputs}, outputs={self.outputs}, code_in_loop={self.code_in_loop}, comp={self.comp})"


class HLSConfig:
    def __init__(self, header_name: str, source_name: str, top_name: str) -> None:
        self.header_name = header_name
        self.source_name = source_name
        self.top_name = top_name
        self.includes = []
        self.defines = []  # TODO: MAX_NUM
        self.structs = {}
        self.functions = {}
        self.PARTITION_FACTOR = 16
        self.STREAM_DEPTH = 4
        self.MAX_NUM = 256
        self.BUFFER_LENGTH = 4

    def __repr__(self) -> str:
        return f"HLSConfig(header_name={self.header_name}, source_name={self.source_name})"

    class ReduceSubFunc:
        def __init__(self, name: str, start_ports: List[dfir.Port], end_ports: List[dfir.Port]) -> None:
            self.name = name
            self.start_ports = start_ports
            self.nxt_ports = start_ports
            self.end_ports = end_ports
            self.satisfieds = []
            self.sub_funcs: List[HLSFunction] = []

        def check_comp(self, comp: dfir.Component, sub_func: HLSFunction) -> bool:
            if not any(p in self.nxt_ports for p in comp.in_ports):
                return False
            assert all(
                p in self.nxt_ports or isinstance(p.connection.parent, dfir.ConstantComponent)
                for p in comp.in_ports
            )
            assert not isinstance(comp, dfir.ReduceComponent)
            self.sub_funcs.append(sub_func)
            self.nxt_ports = [p for p in self.nxt_ports if p not in comp.in_ports]
            for p in comp.out_ports:
                if p in self.end_ports:
                    self.satisfieds.append(p)
                else:
                    assert p.connected
                    self.nxt_ports.append(p.connection)
            self.nxt_ports = list(set(self.nxt_ports))
            return True

        def check_satisfied(self) -> bool:
            return self.satisfieds == self.end_ports

        def __repr__(self) -> str:
            return f"HLSConfig.ReduceSubFunc(name={self.name}, start_ports={self.start_ports}, end_ports={self.end_ports}, satisfieds={self.satisfieds}, sub_funcs={self.sub_funcs})"

        @classmethod
        def from_reduce(cls, comp: dfir.ReduceComponent) -> Tuple[
            Tuple[
                HLSConfig.ReduceSubFunc,
                HLSConfig.ReduceSubFunc,
                HLSConfig.ReduceSubFunc,
            ],
            Tuple[str, str, str],
        ]:
            reduce_key_func_name = f"{comp.name}_key_sub_func"
            reduce_transform_func_name = f"{comp.name}_transform_sub_func"
            reduce_unit_func_name = f"{comp.name}_unit_sub_func"
            key_func = cls(
                name=reduce_key_func_name,
                start_ports=[comp.get_port("o_reduce_key_in").connection],
                end_ports=[comp.get_port("i_reduce_key_out").connection],
            )
            transform_func = cls(
                name=reduce_transform_func_name,
                start_ports=[comp.get_port("o_reduce_transform_in").connection],
                end_ports=[comp.get_port("i_reduce_transform_out").connection],
            )
            unit_func = cls(
                name=reduce_unit_func_name,
                start_ports=[
                    comp.get_port("o_reduce_unit_start_0").connection,
                    comp.get_port("o_reduce_unit_start_1").connection,
                ],
                end_ports=[comp.get_port("i_reduce_unit_end").connection],
            )
            return (key_func, transform_func, unit_func), (
                reduce_key_func_name,
                reduce_transform_func_name,
                reduce_unit_func_name,
            )

    def generate_hls_code(self, global_graph, comp_col: dfir.ComponentCollection) -> str:
        def gen_read_name(name: str, port: dfir.Port, is_sub_func: bool = False) -> str:
            if is_sub_func:
                return name
            return f"{name}.read()"

        dt_manager = HLSDataTypeManager(global_graph.node_properties, global_graph.edge_properties)
        header_code = ""
        source_code = ""
        top_func_def = f"void {self.top_name}(\n"
        top_func_sub_funcs = []
        assert comp_col.inputs == []
        start_ports = []
        end_ports = comp_col.outputs
        constants_from_ports = {}
        source_code += f'#include "{self.header_name}"\n\n'
        source_code += "using namespace hls;\n\n"
        source_code_funcs_part = ""
        reduce_sub_funcs: List[HLSConfig.ReduceSubFunc] = []
        for comp in comp_col.topo_sort():
            assert not isinstance(comp, dfir.PlaceholderComponent)
            if isinstance(comp, dfir.UnusedEndMarkerComponent):
                pass
            elif isinstance(comp, dfir.ConstantComponent):
                assert comp.out_ports[0].connection not in constants_from_ports
                constants_from_ports[comp.out_ports[0].connection] = "{ " + f"{comp.value}" + ", false }"
            elif isinstance(comp, dfir.IOComponent):
                assert comp.io_type == dfir.IOComponent.IOType.INPUT
                for port in comp.ports:
                    assert port.port_type == dfir.PortType.OUT
                    start_ports.append(port.connection)
                    top_func_def += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.connection.unique_name},\n"
            elif isinstance(comp, dfir.ReduceComponent):
                sub_funcs, sub_func_names = HLSConfig.ReduceSubFunc.from_reduce(comp)
                port2type = {port.name: dt_manager.from_dfir_type(port.data_type) for port in comp.ports}
                reduce_sub_funcs.extend(sub_funcs)
                (
                    reduce_key_func_name,
                    reduce_transform_func_name,
                    reduce_unit_func_name,
                ) = sub_func_names
                reduce_pre_func, reduce_unit_func = comp.to_hls_list(
                    func_key_name=reduce_key_func_name,
                    func_transform_name=reduce_transform_func_name,
                    func_unit_name=reduce_unit_func_name,
                )
                reduce_pre_func_str = f"static void {reduce_pre_func.name}(\n"
                in_port = comp.get_port("i_0")
                input_type = in_port.data_type
                key_out_type = comp.get_port("i_reduce_key_out").data_type
                accumulate_type = comp.get_port("i_reduce_unit_end").data_type
                reduce_pre_func_str += "".join(
                    [
                        f"    stream<{dt_manager.from_dfir_type(input_type)}> &i_0,\n"
                        f"    stream<{dt_manager.from_dfir_type(key_out_type)}> &intermediate_key,\n"
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &intermediate_transform\n"
                        # "    uint32_t input_length\n"
                        ") {\n",
                        f"    LOOP_{reduce_pre_func.name}:\n",
                        "    while (true) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )

                call_regex = r"#call:([\w_,]+)#"
                call_once_regex = r"#call_once:([\w_,]+)#"

                def manage_call(line: str) -> str:
                    i = 0
                    while i < len(line) and line[i] == " ":
                        i += 1
                    match = re.search(call_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return " " * i + f"{func_name}({', '.join(args)});"
                    match = re.search(call_once_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return " " * i + f"{func_name}({', '.join(args)});"
                    return line

                for line in reduce_pre_func.code_in_loop:
                    # replace #type# and #read#, only i_0 in reduce_pre_func
                    for port, p_type in port2type.items():
                        line = line.replace(f"#type:{port}#", p_type)
                    line = line.replace("#end_flag_val#", "end_flag_val")
                    if in_port in constants_from_ports:
                        line = line.replace(f"#read:i_0#", f"{constants_from_ports[in_port]}")
                    else:
                        line = line.replace(f"#read:i_0#", f"{gen_read_name('i_0', in_port)}")
                    line = manage_call(line)
                    reduce_pre_func_str += f"        {line}\n"
                reduce_pre_func_str += "        if (end_flag_val) break;\n"
                reduce_pre_func_str += "    }\n"
                reduce_pre_func_str += "}\n"
                intermediate_key_port = dfir.Port(
                    "o_intermediate_key", dfir.EmptyNode(output_type=key_out_type)
                )
                intermediate_key_i_port = dfir.Port(
                    "i_intermediate_key", dfir.EmptyNode(input_type=key_out_type)
                )
                intermediate_key_port.connect(intermediate_key_i_port)
                intermediate_transform_port = dfir.Port(
                    "o_intermediate_transform",
                    dfir.EmptyNode(output_type=accumulate_type),
                )
                intermediate_transform_i_port = dfir.Port(
                    "i_intermediate_transform",
                    dfir.EmptyNode(input_type=accumulate_type),
                )
                intermediate_transform_port.connect(intermediate_transform_i_port)
                reduce_pre_func.params = [
                    ("i_0", dt_manager.from_dfir_type(input_type), in_port, True),
                    (
                        "intermediate_key",
                        dt_manager.from_dfir_type(key_out_type),
                        intermediate_key_port,
                        False,
                    ),
                    (
                        "intermediate_transform",
                        dt_manager.from_dfir_type(accumulate_type),
                        intermediate_transform_port,
                        False,
                    ),
                    # ("input_length", True),
                ]

                source_code_funcs_part += reduce_pre_func_str + "\n\n"

                # handle reduce_unit_func
                reduce_key_struct = dt_manager.from_dfir_type(
                    dftype.TupleType(
                        [
                            accumulate_type,
                            dftype.BoolType(),
                        ]
                    ),
                    sub_names=["data", "valid"],
                )
                codes_before_loop = [
                    line.replace("#reduce_key_struct#", reduce_key_struct).replace(
                        "#partition_factor#", str(self.PARTITION_FACTOR)
                    )
                    for line in reduce_unit_func.code_before_loop
                ]
                reduce_unit_func_str = "".join(
                    [
                        f"static void {reduce_unit_func.name}(\n",
                        f"    stream<{dt_manager.from_dfir_type(key_out_type)}> &intermediate_key,\n",
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &intermediate_transform,\n",
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &o_0\n",
                        # "    uint32_t &output_length,\n",
                        # "    uint32_t input_length\n",
                        ") {\n",
                    ]
                    + [f"    {line}\n" for line in codes_before_loop]
                    + [
                        f"    LOOP_{reduce_unit_func.name}:\n",
                        "    while (true) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )
                reduce_unit_func.params = [
                    (
                        "intermediate_key",
                        dt_manager.from_dfir_type(key_out_type),
                        intermediate_key_i_port,
                        True,
                    ),
                    (
                        "intermediate_transform",
                        dt_manager.from_dfir_type(accumulate_type),
                        intermediate_transform_i_port,
                        True,
                    ),
                    (
                        "o_0",
                        port2type["o_0"],
                        reduce_unit_func.comp.get_port("o_0"),
                        False,
                    ),
                    # ("output_length", False),
                    # ("input_length", True),
                ]
                for line in reduce_unit_func.code_in_loop:
                    for inter_name, inter_p in [
                        ("intermediate_key", intermediate_key_i_port),
                        ("intermediate_transform", intermediate_transform_i_port),
                        (
                            "reduce_unit_stream_out",
                            reduce_unit_func.comp.get_port("i_reduce_unit_end"),
                        ),
                    ]:
                        line = line.replace(
                            f"#read:{inter_name}#",
                            f"{gen_read_name(inter_name, inter_p)}",
                        )
                    for port, p_type in port2type.items():
                        line = line.replace(f"#type:{port}#", p_type)
                        line = line.replace("#reduce_key_struct#", reduce_key_struct)
                        line = line.replace("#end_flag_val#", "end_flag_val")
                        # if f"#may_ele:{port}#" in line:
                        #     print(reduce_unit_func.comp.get_port(port))
                        #     print(line)
                        line = line.replace(
                            f"#may_ele:{port}#",
                            (".ele" if reduce_unit_func.comp.get_port(port).data_type.is_basic_type else ""),
                        )
                        # look for #cmpeq:type_port,a,b# and if type is edge, assert False, if node, use a.id == b.id, otherwise use a == b
                        line = line.replace(
                            f"#read:{port}#",
                            f"{gen_read_name(port, reduce_unit_func.comp.get_port(port))}",
                        )
                        cmp_regex = r"#cmpeq:([\w_]+),([\w_\.]+),([\w_\.]+)#"
                        match = re.search(cmp_regex, line)
                        if match and port == match.group(1):
                            _, cmp_a, cmp_b = match.groups()
                            if isinstance(
                                comp.get_port(port).data_type,
                                (dftype.BoolType, dftype.IntType),
                            ):
                                line = line.replace(match.group(0), f"{cmp_a} == {cmp_b}")
                            elif comp.get_port(port).data_type == dftype.SpecialType("node"):
                                line = line.replace(match.group(0), f"{cmp_a}.id.ele == {cmp_b}.id.ele")
                            else:
                                assert False, "Not supported type comparing"
                        line = manage_call(line)
                    # for write
                    if "#write" in line:
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space = " " * (i + 8)
                        potential_match_str = line[i:]
                        pattern = re.compile(
                            r"^#write:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$"
                        )
                        match = pattern.match(potential_match_str)
                        pattern_nostream = re.compile(
                            r"^#write_nostream:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$"
                        )
                        match_nostream = pattern_nostream.match(potential_match_str)
                        assert (
                            match or match_nostream
                        ), f"Error: Line '{line.strip()}' include '#write' but with wrong format."
                        nostream = False
                        if match_nostream:
                            match = match_nostream
                            nostream = True

                        tgt_port_name, ori_var, tgt_outer_type = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        line = "" if not nostream else (indent_space + f"{tgt_outer_type} {tgt_port_name};\n")
                        line += indent_space + r"{" + "\n"
                        line += (
                            indent_space
                            + f"    {tgt_inner_type}_to_{tgt_outer_type}({ori_var}, tmp_{tgt_outer_type}_var, true);\n"
                        )
                        line += (
                            (indent_space + f"    {tgt_port_name}.write(tmp_{tgt_outer_type}_var);\n")
                            if not nostream
                            else (indent_space + f"    {tgt_port_name} = tmp_{tgt_outer_type}_var;\n")
                        )
                        line += indent_space + r"}" + "\n"
                        reduce_unit_func_str += line
                    elif "#peel" in line:
                        pattern = re.compile(r"^#peel:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$")
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space = " " * (i + 8)
                        potential_match_str = line[i:]
                        match = pattern.match(potential_match_str)
                        assert match, f"Error: Line '{line.strip()}' include '#peel' but with wrong format."

                        tgt_port_name, ori_var, new_var = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        reduce_unit_func_str += (
                            indent_space + f"{tgt_outer_type}_to_{tgt_inner_type}({ori_var}, {new_var});\n"
                        )
                    else:
                        reduce_unit_func_str += f"        {line}\n"
                reduce_unit_func_str += "        if (end_flag_val) break;\n"
                reduce_unit_func_str += "    }\n"
                for line in reduce_unit_func.code_after_loop:
                    # line = line.replace("#output_length#", "output_length")
                    for port, p_type in port2type.items():
                        line = line.replace(
                            f"#may_ele:{port}#",
                            (".ele" if reduce_unit_func.comp.get_port(port).data_type.is_basic_type else ""),
                        )
                    if "#write" in line:
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space = " " * (i + 4)
                        potential_match_str = line[i:]
                        pattern = re.compile(r"^#write:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$")
                        pattern_noend = re.compile(r"^#write_noend:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$")
                        pattern_notrans = re.compile(
                            r"^#write_notrans:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$"
                        )
                        match = pattern.match(potential_match_str)
                        match_noend = pattern_noend.match(potential_match_str)
                        match_notrans = pattern_notrans.match(potential_match_str)
                        assert (
                            match or match_noend or match_notrans
                        ), f"Error2: Line '{line.strip()}' include '#write' but with wrong format."
                        ending = "false" if match_noend else "true"
                        do_trans = False if match_notrans else True

                        if match_noend:
                            match = match_noend
                        elif match_notrans:
                            match = match_notrans
                        tgt_port_name, ori_var = match.group(1), match.group(2)
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        line = indent_space + r"{" + "\n"
                        line += (
                            (
                                indent_space
                                + f"    {tgt_inner_type}_to_{tgt_outer_type}({ori_var}, tmp_{tgt_outer_type}_var, {ending});\n"
                            )
                            if do_trans
                            else ""
                        )
                        line += indent_space + f"    {tgt_port_name}.write(tmp_{tgt_outer_type}_var);\n"
                        line += indent_space + r"}" + "\n"
                        reduce_unit_func_str += line
                    else:
                        reduce_unit_func_str += f"    {line}\n"
                reduce_unit_func_str += "}\n"
                source_code_funcs_part += reduce_unit_func_str + "\n\n"
                top_func_sub_funcs.extend([reduce_pre_func, reduce_unit_func])
            else:
                func = comp.to_hls()
                # check if any port connected to EndMarker
                unused_ports = []
                for port in comp.ports:
                    if port.connected and isinstance(port.connection.parent, dfir.UnusedEndMarkerComponent):
                        unused_ports.append(port.name)
                is_sub_func = False
                # check & add for reduce sub functions
                for sub_func in reduce_sub_funcs:
                    is_sub_func |= sub_func.check_comp(comp, func)
                # generate function str
                func_str = f"static void {func.name}(\n"
                # type, read, output_length, opt_type
                port2type = {}
                for port in func.comp.ports:
                    port2type[port.name] = dt_manager.from_dfir_type(port.data_type)
                    if port in constants_from_ports or port.name in unused_ports:
                        continue
                    if not is_sub_func:
                        func_str += f"    stream<{port2type[port.name]}> &{port.name},\n"
                    else:
                        func_str += f"    {port2type[port.name]} &{port.name},\n"
                    func.params.append(
                        (
                            port.unique_name,
                            port2type[port.name],
                            port,
                            port.port_type == dfir.PortType.IN,
                        )
                    )
                # delete last ","
                if func_str[-2:] == ",\n":
                    func_str = func_str[:-2] + "\n"

                # if any("#output_length#" in line for line in func.code_in_loop):
                #     func_str += "    uint32_t &output_length,\n"
                #     func.params.append(("output_length", False))
                # func_str += "    uint32_t input_length\n"
                # func.params.append(("input_length", True))
                func_str += ")"

                source_code += func_str + ";\n\n"
                func_str += " {\n"
                if is_sub_func:
                    func_str += "#pragma HLS INLINE\n"
                has_end_flag = False

                def manage_line(line: str, indent: int) -> str:
                    if "write" in line:
                        print(func.name, line, unused_ports)
                    # for unused ports
                    for port_name in unused_ports:
                        if (
                            f"#read:{port_name}#" in line
                            or f"{port_name}.write" in line
                            or f"#write:{port_name}" in line
                        ):
                            return ""
                    # # output_length
                    # line = line.replace(f"#output_length#", "output_length")
                    # end flag val
                    nonlocal has_end_flag
                    if "#end_flag_val#" in line:
                        has_end_flag = True
                    line = line.replace("#end_flag_val#", "end_flag_val")
                    # type declaration & read
                    for port, type in port2type.items():
                        line = line.replace(f"#type:{port}#", type)
                        line = line.replace(f"#type_inner:{port}#", type[6:])
                        if comp.get_port(port) in constants_from_ports:
                            line = line.replace(
                                f"#read:{port}#",
                                f"{constants_from_ports[comp.get_port(port)]}",
                            )
                        else:
                            line = line.replace(
                                f"#read:{port}#",
                                f"{gen_read_name(port, comp.get_port(port), is_sub_func)}",
                            )
                        line = line.replace(f"#opt_type:{port}#", port2type[port][6:])

                        port_type = comp.get_port(port).data_type
                        if isinstance(port_type, dfir.ArrayType):
                            port_type = port_type.type_
                        line = line.replace(
                            f"#may_ele:{port}#",
                            (".ele" if port_type.is_basic_type else ""),
                        )
                    indent_space = " " * 4 * indent
                    # for write
                    if "#write" in line:
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space += " " * i
                        potential_match_str = line[i:]
                        pattern = re.compile(r"^#write:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$")
                        pattern_notrans = re.compile(
                            r"^#write_notrans:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$"
                        )
                        match = pattern.match(potential_match_str)
                        match_notrans = pattern_notrans.match(potential_match_str)
                        assert (
                            match or match_notrans
                        ), f"Error2: Line '{line.strip()}' include '#write' but with wrong format."
                        do_trans = False if match_notrans else True

                        if match_notrans:
                            match = match_notrans

                        tgt_port_name, ori_var = match.group(1), match.group(2)
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        line = indent_space + r"{" + "\n"
                        line += (
                            (
                                indent_space
                                + f"    {tgt_inner_type}_to_{tgt_outer_type}({ori_var}, tmp_{tgt_outer_type}_var, end_flag_val);\n"
                            )
                            if do_trans
                            else ""
                        )
                        target_final_var = f"tmp_{tgt_outer_type}_var" if do_trans else ori_var
                        if is_sub_func:
                            line += indent_space + f"    {tgt_port_name} = {target_final_var};\n"
                        else:
                            line += indent_space + f"    {tgt_port_name}.write({target_final_var});\n"
                        line += indent_space + r"}" + "\n"
                        return line
                    elif "#peel" in line:
                        pattern = re.compile(r"^#peel:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$")
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space += " " * i
                        potential_match_str = line[i:]
                        match = pattern.match(potential_match_str)
                        assert match, f"Error: Line '{line.strip()}' include '#peel' but with wrong format."

                        tgt_port_name, ori_var, new_var = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        return indent_space + f"{tgt_outer_type}_to_{tgt_inner_type}({ori_var}, {new_var});\n"
                    else:
                        return indent_space + line + "\n"

                for line in func.code_before_loop:
                    func_str += f"{manage_line(line, 1)}"
                # sub func no loop
                if not is_sub_func:
                    func_str += f"    LOOP_{func.name}:\n"
                    # func_str += "    for (uint32_t i = 0; i < input_length; i++) {\n"
                    func_str += "    while (true) {\n"
                    func_str += "#pragma HLS PIPELINE\n"
                for line in func.code_in_loop:
                    func_str += f"{manage_line(line, 2)}"
                assert has_end_flag
                if not is_sub_func:
                    func_str += "        if (end_flag_val) break;\n"
                    func_str += "    }\n"
                for line in func.code_after_loop:
                    func_str += f"{manage_line(line, 1)}"
                func_str += "}\n"
                source_code_funcs_part += func_str + "\n\n"
                top_func_sub_funcs.append(func)

        # manage top function end ports & input len
        for port in end_ports:
            top_func_def += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.unique_name},\n"
        # top_func_def += "    uint32_t input_length\n"
        if top_func_def[-2:] == ",\n":
            top_func_def = top_func_def[:-2] + "\n"
        top_func_def += ")"

        # manage reduce sub functions
        assert all(sub_func.check_satisfied() for sub_func in reduce_sub_funcs)
        for sub_func in reduce_sub_funcs:
            sub_func_code = f"static void {sub_func.name}(\n"
            for port in sub_func.start_ports + sub_func.end_ports:
                # sub_func_code += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.unique_name},\n"
                sub_func_code += f"    {dt_manager.from_dfir_type(port.data_type)} &{port.unique_name},\n"
            # if any(sub_sub_func.change_length for sub_sub_func in sub_func.sub_funcs):
            #     sub_func_code += "    uint32_t &output_length,\n"
            # sub_func_code += "    uint32_t input_length\n"
            if sub_func_code[-2:] == ",\n":
                sub_func_code = sub_func_code[:-2] + "\n"
            sub_func_code += ") {\n#pragma HLS INLINE\n"
            sub_func_code += self.generate_sub_func_code(
                sub_func.start_ports, sub_func.end_ports, sub_func.sub_funcs, True
            )
            top_func_sub_funcs = [
                cur_sub_func for cur_sub_func in top_func_sub_funcs if cur_sub_func not in sub_func.sub_funcs
            ]
            sub_func_code += "}\n"
            source_code += sub_func_code + "\n\n"

        # add functions part
        source_code += source_code_funcs_part

        # manage structure defines
        all_defines = dt_manager.get_all_defines()
        header_code += "\n\n".join(all_defines)

        # add top module
        header_code += "\n\n" + top_func_def + ";"
        top_func_code = top_func_def + " {\n"
        top_func_code += "#pragma HLS dataflow\n"
        # if any(sub_sub_func.change_length for sub_sub_func in top_func_sub_funcs):
        #     # top_func_code += "    uint32_t output_length = input_length;\n"
        #     print(end_ports)
        top_func_code += self.generate_sub_func_code(start_ports, end_ports, top_func_sub_funcs)
        top_func_code += "}\n"
        source_code += top_func_code + "\n\n"

        # add start & end define for header
        header_name_for_define = f'__{self.header_name.upper().replace(".", "_")}__'
        header_code = (
            f"#ifndef {header_name_for_define}"
            + f"\n#define {header_name_for_define}\n\n"
            + "#include <stdint.h>\n#include <ap_int.h>\n#include <hls_stream.h>\n\n"
            + "using namespace hls;\nusing namespace std;\n\n"
            + f"#define MAX_NUM {self.MAX_NUM}\n"
            + f"#define L {self.BUFFER_LENGTH}\n\n"
            + header_code
            + "\n\n"
            + f"#endif // {header_name_for_define}\n"
        )

        return header_code, source_code

    def generate_sub_func_code(
        self,
        start_ports: List[dfir.Port],
        end_ports: List[dfir.Port],
        functions: List[HLSFunction],
        is_sub_sub_func: bool = False,
    ) -> str:
        port2var_name = {}
        adding_codes = ""
        for sub_sub_func in functions:
            call_code = f"    {sub_sub_func.name}(\n"
            call_params = []
            for param in sub_sub_func.params:
                port_name, port_type, cur_port, is_in = param
                if is_in:
                    if any(cur_port == st_p for st_p in start_ports):
                        sub_sub_func_var_name = f"{cur_port.unique_name}"
                    else:
                        sub_sub_func_var_name = port2var_name[cur_port.connection]
                    call_params.append(sub_sub_func_var_name)
                else:
                    if any(cur_port == ed_p for ed_p in end_ports):
                        sub_sub_func_var_name = f"{cur_port.unique_name}"
                    else:
                        sub_sub_func_var_name = f"{sub_sub_func.name}_{cur_port.unique_name}"
                        port2var_name[cur_port] = sub_sub_func_var_name
                        adding_codes += (
                            (
                                f"    stream<{port_type}> {sub_sub_func_var_name};\n"
                                f"    #pragma HLS STREAM variable={sub_sub_func_var_name} depth={self.STREAM_DEPTH}\n"
                            )
                            if not is_sub_sub_func
                            else (f"    {port_type} {sub_sub_func_var_name};\n")
                        )
                    call_params.append(sub_sub_func_var_name)

            call_code += ",\n".join(f"        {param}" for param in call_params)
            call_code += "\n    );\n"
            adding_codes += call_code
        return adding_codes


global_hls_config = HLSConfig(
    header_name="graphyflow.h",
    source_name="graphyflow.cpp",
    top_name="graphyflow",
)

```

`CCFSys2025_GraphyFlow/graphyflow/lambda_func.py`:

```py
import inspect
from warnings import warn
import graphyflow.dataflow_ir as dfir
from typing import Dict, List, Tuple, Any, Callable, Optional


def lambda_min(tracer1, tracer2):
    return _lambda_binop(tracer1, tracer2, "min", min)


def lambda_max(tracer1, tracer2):
    return _lambda_binop(tracer1, tracer2, "max", max)


def _lambda_binop(tracer1, tracer2, op, default_func: Callable):
    if isinstance(tracer1, Tracer) and isinstance(tracer2, Tracer):
        return Tracer(node_type="operation", operator=op, parents=[tracer1, tracer2])
    elif isinstance(tracer1, Tracer):
        constant_tracer = Tracer(node_type="constant", value=tracer2)
        return Tracer(node_type="operation", operator=op, parents=[tracer1, constant_tracer])
    elif isinstance(tracer2, Tracer):
        constant_tracer = Tracer(node_type="constant", value=tracer1)
        return Tracer(node_type="operation", operator=op, parents=[constant_tracer, tracer2])
    else:
        return default_func(tracer1, tracer2)


class Tracer:
    _tracer_id = 0  # for generating unique id

    def __init__(
        self,
        name=None,
        node_type="input",
        attr_name=None,
        operator=None,
        parents=None,
        value=None,
        pseudo_element=None,
    ):
        self._id = Tracer._tracer_id
        Tracer._tracer_id += 1
        self._name = name
        self._node_type = node_type
        self._attr_name = attr_name
        self._operator = operator
        self._parents = parents or []
        self._value = value
        self._pseudo_element = pseudo_element

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Tracer(node_type="attr", attr_name=name, parents=[self])

    def __getitem__(self, index):
        assert type(index) == int
        return Tracer(node_type="idx", attr_name=index, parents=[self])

    def _bin_op(self, other, op, reverse=False):
        # Non-Tracer -> Constant
        if not isinstance(other, Tracer):
            other = Tracer(node_type="constant", value=other)
        return Tracer(
            node_type="operation",
            operator=op,
            parents=[other, self] if reverse else [self, other],
        )

    def __add__(self, other):
        return self._bin_op(other, "+")

    def __radd__(self, other):
        return self._bin_op(other, "+", reverse=True)

    def __sub__(self, other):
        return self._bin_op(other, "-")

    def __rsub__(self, other):
        return self._bin_op(other, "-", reverse=True)

    def __mul__(self, other):
        return self._bin_op(other, "*")

    def __rmul__(self, other):
        return self._bin_op(other, "*", reverse=True)

    def __truediv__(self, other):
        return self._bin_op(other, "/")

    def __rtruediv__(self, other):
        return self._bin_op(other, "/", reverse=True)

    def __lt__(self, other):
        return self._bin_op(other, "<")

    def __gt__(self, other):
        return self._bin_op(other, ">")

    def __le__(self, other):
        return self._bin_op(other, "<=")

    def __ge__(self, other):
        return self._bin_op(other, ">=")

    def __eq__(self, other):
        return self._bin_op(other, "==")

    def __ne__(self, other):
        return self._bin_op(other, "!=")

    def to_dict(self):
        return {
            k: v
            for k, v in [
                ("id", self._id),
                ("parents", [p._id for p in self._parents]),
                ("type", self._node_type),
                ("name", self._name),
                ("attr", self._attr_name),
                ("operator", self._operator),
                ("value", self._value),
                ("pseudo_element", self._pseudo_element),
            ]
            if v is not None
        }


def parse_lambda(lambda_func):
    try:
        sig = inspect.signature(lambda_func)
        num_params = len(sig.parameters)
    except:
        raise RuntimeError("Lambda function analyze failed.")

    inputs = [Tracer(name=f"arg{i}", node_type="input") for i in range(num_params)]

    try:
        result = lambda_func(*inputs)
    except Exception as e:
        raise RuntimeError(f" {str(e)}")

    outputs = result if isinstance(result, (tuple, list)) else [result]
    if isinstance(outputs, tuple):
        outputs = list(outputs)

    all_nodes = []
    visited = set()

    def traverse(node):
        if node._id in visited:
            return
        visited.add(node._id)
        all_nodes.append(node)
        for parent in node._parents:
            traverse(parent)

    for i in range(len(outputs)):
        if not isinstance(outputs[i], Tracer):
            outputs[i] = Tracer(value=outputs[i], node_type="constant")
        traverse(outputs[i])

    for node in inputs:
        if node._id not in visited:
            all_nodes.append(node)

    edges = [(p._id, node._id) for node in all_nodes for p in node._parents]

    return {
        "nodes": {n._id: n.to_dict() for n in all_nodes},
        "edges": edges,
        "input_ids": [n._id for n in inputs],
        "output_ids": [n._id for n in outputs],
    }


def format_lambda(lambda_dict):
    nodes = lambda_dict["nodes"]
    node_strs = {}

    node_type_formats = {
        "input": lambda info: f"Input ({info.get('name', 'unnamed')})",
        "operation": lambda info: f"Operation ({info.get('operator', '?')})",
        "attr": lambda info: f"Attribute (.{info.get('attr', '?')})",
        "idx": lambda info: f"Index ([{info.get('attr', '?')}])",
        "constant": lambda info: f"Constant ({info.get('value', '?')})",
        "pseudo": lambda info: f"{info.get('pseudo_element', '?')}",
    }

    for node_id, node_info in sorted(nodes.items()):
        node_str = f"Node {node_id}: "
        node_type = node_info.get("type", "unknown")
        if node_type in node_type_formats:
            node_str += node_type_formats[node_type](node_info)
        else:
            node_str += str(node_info)
        if node_id in lambda_dict["input_ids"]:
            node_str += " [INPUT]"
        if node_id in lambda_dict["output_ids"]:
            node_str += " [OUTPUT]"
        node_strs[node_id] = node_str

    result = ["Lambda Repr: "]

    visited = set()

    if len(lambda_dict["edges"]) == 0:
        result.extend(f"  {node_str}" for node_str in node_strs.values())
    else:
        max_src_len = max(len(node_strs[src]) for src, _ in lambda_dict["edges"])
        for src, dst in sorted(lambda_dict["edges"]):
            padding = " " * (max_src_len - len(node_strs[src]) + 2)
            result.append(f"  {node_strs[src]}{padding}→ {node_strs[dst]}")
            visited.add(src)
            visited.add(dst)
    result.append("Unconnected Nodes: {")
    for node_id in sorted(node_strs.keys()):
        if node_id not in visited:
            result.append(f"  {node_strs[node_id]}")
    result.append("}")

    result.append(f"Input Nodes: {', '.join(map(str, lambda_dict['input_ids']))}")
    result.append(f"Output Nodes: {', '.join(map(str, lambda_dict['output_ids']))}")

    return "\n".join(result)


def lambda_to_dfir(
    lambda_dict: Dict[str, Any],
    input_types: List[dfir.DfirType],
    node_prop: Optional[Dict[str, Any]],
    edge_prop: Optional[Dict[str, Any]],
    scatter_outputs: bool = True,
) -> dfir.ComponentCollection:
    assert len(input_types) == len(lambda_dict["input_ids"])
    assert all(isinstance(t, dfir.DfirType) for t in input_types)
    is_parallel = all(isinstance(t, dfir.ArrayType) for t in input_types)

    def translate_constant_val(node, pre_o_types):
        assert node["value"] is not None and type(node["value"]) in [bool, int, float]
        value_dfir_dict = {
            int: dfir.IntType(),
            bool: dfir.BoolType(),
            float: dfir.FloatType(),
        }
        dfir_type = value_dfir_dict[type(node["value"])]
        if is_parallel:
            dfir_type = dfir.ArrayType(dfir_type)
        return dfir.ConstantComponent(dfir_type, node["value"])

    def translate_bin_op(node, pre_o_types):
        assert node["operator"] in [
            "+",
            "-",
            "*",
            "/",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            "min",
            "max",
        ]
        assert (
            len(pre_o_types) == 2 and pre_o_types[0] == pre_o_types[1]
        ), f"{pre_o_types} should be 2 equal types"
        op_dfir_dict = {
            "+": dfir.BinOp.ADD,
            "-": dfir.BinOp.SUB,
            "*": dfir.BinOp.MUL,
            "/": dfir.BinOp.DIV,
            "<": dfir.BinOp.LT,
            "<=": dfir.BinOp.LE,
            ">": dfir.BinOp.GT,
            ">=": dfir.BinOp.GE,
            "==": dfir.BinOp.EQ,
            "!=": dfir.BinOp.NE,
            "min": dfir.BinOp.MIN,
            "max": dfir.BinOp.MAX,
        }
        return dfir.BinOpComponent(op_dfir_dict[node["operator"]], pre_o_types[0])

    def translate_attr(node, pre_o_types):
        assert pre_o_types[0] in [
            dfir.SpecialType("node"),
            dfir.SpecialType("edge"),
        ] or (
            isinstance(pre_o_types[0], dfir.ArrayType)
            and pre_o_types[0].type_ in [dfir.SpecialType("node"), dfir.SpecialType("edge")]
        )
        if pre_o_types[0] == dfir.SpecialType("node") or (
            isinstance(pre_o_types[0], dfir.ArrayType) and pre_o_types[0].type_ == dfir.SpecialType("node")
        ):
            assert node["attr"] in node_prop, f"{node['attr']} not in {node_prop}"
            return dfir.UnaryOpComponent(
                dfir.UnaryOp.GET_ATTR,
                pre_o_types[0],
                select_index=node["attr"],
                attr_type=node_prop[node["attr"]],
            )
        else:
            assert node["attr"] in edge_prop, f"{node['attr']} not in {edge_prop}"
            return dfir.UnaryOpComponent(
                dfir.UnaryOp.GET_ATTR,
                pre_o_types[0],
                select_index=node["attr"],
                attr_type=edge_prop[node["attr"]],
            )

    translate_dict = {
        "input": lambda node, pre_o_types: dfir.PlaceholderComponent(pre_o_types[0]),
        "attr": translate_attr,
        "idx": lambda node, pre_o_types: dfir.UnaryOpComponent(
            dfir.UnaryOp.SELECT, pre_o_types[0], node["attr"]
        ),
        "constant": translate_constant_val,
        "operation": translate_bin_op,
    }
    nodes, edges = lambda_dict["nodes"], lambda_dict["edges"]
    dfir_nodes = {}
    max_nid = max(nodes.keys())
    in_degree = {nid: len(list(dst for _, dst in edges if dst == nid)) for nid in nodes}
    start_queue = [nid for nid, deg in in_degree.items() if deg == 0]
    # delete the deg==0 nodes from in_degree
    for nid in start_queue:
        if nid in in_degree:
            del in_degree[nid]
    constants = [nid for nid in start_queue if nodes[nid]["type"] == "constant"]
    assert len(input_types) == len(start_queue) - len(constants)
    start_queue = [nid for nid in start_queue if nid not in constants]
    queue = []
    node_tmp_datas = {nid: [nid, [], {}] for nid in nodes.keys()}
    for nid in constants:
        queue.append([nid, [], {}])
    for name_id in range(len(input_types)):
        arg_name = f"arg{name_id}"
        target = [nid for nid in start_queue if nodes[nid]["name"] == arg_name]
        assert len(target) == 1
        queue.append([target[0], [input_types[name_id]], {}])  # node, prev_type, parent_ports
    both_in_out_nids = [nid for nid in lambda_dict["input_ids"] if nid in lambda_dict["output_ids"]]
    both_in_out_ports = []
    both_in_out_ports_waitlist = {}
    while queue:
        nid, pre_o_types, p_ports = queue.pop(0)
        node_type = nodes[nid]["type"]
        dfir_nodes[nid] = translate_dict[node_type](nodes[nid], pre_o_types)
        if nid in both_in_out_nids:
            for p in dfir_nodes[nid].ports:
                both_in_out_ports.append(p)
                both_in_out_ports_waitlist[p] = []

        for my_port_name, p_port in p_ports.items():
            if p_port in both_in_out_ports:
                both_in_out_ports_waitlist[p_port].append((nid, my_port_name))
            elif p_port.connected:
                copy_comp = dfir.CopyComponent(p_port.data_type)
                dfir_nodes[max_nid + 1] = copy_comp
                max_nid += 1
                new_port = p_port.copy(copy_comp)
                p_ports[my_port_name] = new_port
        p_ports = {mp_name: p_port for mp_name, p_port in p_ports.items() if not p_port in both_in_out_ports}
        dfir_nodes[nid].connect(p_ports)
        out_type = dfir_nodes[nid].output_type
        succ_node_ids = [dst for src, dst in edges if src == nid]
        for succ_nid in succ_node_ids:
            node_tmp_datas[succ_nid][1].append(out_type)
            for p in dfir_nodes[nid].out_ports:
                succ_parent_list = nodes[succ_nid]["parents"]
                for in_id, succ_parent_id in enumerate(succ_parent_list):
                    if succ_parent_id == nid:
                        node_tmp_datas[succ_nid][2][f"i_{in_id}"] = p
            in_degree[succ_nid] -= 1
            if in_degree[succ_nid] == 0:
                queue.append(node_tmp_datas[succ_nid])
                del in_degree[succ_nid]
    port_trans_dict = {}
    for p_port, waitings in both_in_out_ports_waitlist.items():
        while len(waitings) > 0:
            nid, my_port_name = waitings.pop()
            copy_comp = dfir.CopyComponent(p_port.data_type)
            dfir_nodes[max_nid + 1] = copy_comp
            max_nid += 1
            target_port = port_trans_dict[p_port] if p_port in port_trans_dict else p_port
            dfir_nodes[nid].connect({my_port_name: target_port})
            new_target_port = target_port.copy(copy_comp)
            port_trans_dict[p_port] = new_target_port
    inputs = sum([dfir_nodes[nid].in_ports for nid in lambda_dict["input_ids"]], [])
    outputs = sum([dfir_nodes[nid].out_ports for nid in lambda_dict["output_ids"]], [])
    for i in range(len(outputs)):
        if outputs[i] in port_trans_dict:
            outputs[i] = port_trans_dict[outputs[i]]
    assert all(not p.connected for p in (inputs + outputs))
    # handle unused input
    all_ports = sum((node.ports for node in dfir_nodes.values()), [])
    all_hanged_ports = [p for p in all_ports if (not p.connected and not p in inputs + outputs)]
    all_hanged_nodes = [
        nid for nid in dfir_nodes if any(p in all_hanged_ports for p in dfir_nodes[nid].ports)
    ]
    all_input_dfir_nodes = [nid for nid in lambda_dict["input_ids"]]
    # only input dfir nodes can be hanged
    assert all(n in all_input_dfir_nodes for n in all_hanged_nodes)
    # add a unused end marker to the hanged nodes
    for nid in all_hanged_nodes:
        dfir_nodes[max_nid + 1] = dfir.UnusedEndMarkerComponent(dfir_nodes[nid].output_type)
        assert len(dfir_nodes[nid].out_ports) == 1
        dfir_nodes[max_nid + 1].connect({"i_0": dfir_nodes[nid].out_ports[0]})
        max_nid += 1
    # handle scatter outputs
    if not scatter_outputs and len(outputs) > 1:
        gather_comp = dfir.GatherComponent(list(map(lambda port: port.data_type, outputs)))
        for i, p in enumerate(outputs):
            gather_comp.ports[i].connect(p)
        dfir_nodes[max_nid + 1] = gather_comp
        max_nid += 1
        outputs = gather_comp.out_ports.copy()
    return dfir.ComponentCollection(list(dfir_nodes.values()), inputs, outputs)


if __name__ == "__main__":
    func = lambda a, b: a.x * a.y + 2 / b
    graph = parse_lambda(func)
    the_dfir = lambda_to_dfir(graph, [dfir.SpecialType("node"), dfir.IntType()])

    if graph:
        print("Nodes:")
        for nid, info in graph["nodes"].items():
            print(f"Node {nid}: {info}")

        print("\nEdges:")
        for src, dst in graph["edges"]:
            print(f"{src} -> {dst}")

        print("\nInput Node(s):", graph["input_ids"])
        print("Output Node(s)", graph["output_ids"])

    print(the_dfir)

```

`CCFSys2025_GraphyFlow/graphyflow/passes.py`:

```py
import graphyflow.dataflow_ir as dfir
from typing import List, Tuple, Set


def delete_placeholder_components_pass(
    comp_col: dfir.ComponentCollection,
) -> dfir.ComponentCollection:
    components_to_keep: List[dfir.Component] = []

    for comp in comp_col.components:
        if isinstance(comp, dfir.PlaceholderComponent):
            ph_input_port = comp.get_port("i_0")
            ph_output_port = comp.get_port("o_0")

            upstream_connected_port = ph_input_port.connection
            downstream_connected_port = ph_output_port.connection

            if upstream_connected_port is not None:
                upstream_connected_port.disconnect()
            else:
                assert ph_input_port in comp_col.inputs
                assert downstream_connected_port is not None, "Empty collection found."
                comp_col.inputs.remove(ph_input_port)
                comp_col.inputs.append(downstream_connected_port)

            if downstream_connected_port is not None:
                downstream_connected_port.disconnect()
            else:
                assert ph_output_port in comp_col.outputs
                assert upstream_connected_port is not None, "Empty collection found."
                comp_col.outputs.remove(ph_output_port)
                comp_col.outputs.append(upstream_connected_port)

            if upstream_connected_port is not None and downstream_connected_port is not None:
                upstream_connected_port.connect(downstream_connected_port)

        else:
            components_to_keep.append(comp)

    comp_col = dfir.ComponentCollection(components_to_keep, comp_col.inputs, comp_col.outputs)
    comp_col.update_ports()

    return comp_col


if __name__ == "__main__":
    from graphyflow.global_graph import *
    from graphyflow.visualize_ir import visualize_components

    g = GlobalGraph(
        properties={
            "node": {"weight": dfir.IntType()},
            "edge": {"e_id": dfir.IntType()},
        }
    )
    nodes = g.add_graph_input("edge")
    src_dst_weight = nodes.map_(map_func=lambda edge: (edge.src.weight, edge.dst.weight, edge))
    # first_reduce = src_dst_weight.reduce_by(
    #     reduce_key=lambda sw, dw, e: sw,
    #     reduce_transform=lambda sw, dw, e: (sw, dw, e),
    #     reduce_method=lambda x, y: (x[0], x[1], x[2]),
    # )
    filtered = src_dst_weight.filter(filter_func=lambda sw, dw, e: sw > dw)
    reduced_result = filtered.reduce_by(
        reduce_key=lambda sw, dw, e: e.dst,
        reduce_transform=lambda sw, dw, e: (sw, e.dst),
        reduce_method=lambda x, y: (x[0] + y[0], x[1]),
    )
    result = reduced_result.map_(map_func=lambda w, dst: (w, dst.weight, dst))

    dfirs = g.to_dfir()
    dfirs[0] = delete_placeholder_components_pass(dfirs[0])
    dot = visualize_components(str(dfirs[0]))
    dot.render("component_graph", view=False, format="png")
    # print(dfirs[0].topo_sort())
    import graphyflow.hls_utils as hls

    header, source = hls.global_hls_config.generate_hls_code(g, dfirs[0])
    import os

    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/graphyflow.h", "w") as f:
        f.write(header)
    with open("output/graphyflow.cpp", "w") as f:
        f.write(source)

```

`CCFSys2025_GraphyFlow/graphyflow/project_generator.py`:

```py
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
from .backend_manager import BackendManager


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)


def generate_project(
    comp_col: ComponentCollection,
    global_graph: Any,
    kernel_name: str,
    output_dir: Path,
    executable_name: str = "host",
    template_dir_override: Optional[Path] = None,
):
    """
    Generates a complete Vitis project directory from a DFG-IR.
    This version includes templating for build and run scripts.
    """
    print(f"--- Starting Project Generation for Kernel '{kernel_name}' ---")

    # 1. 定义路径
    if template_dir_override:
        template_dir = template_dir_override
    else:
        project_root = Path(__file__).parent.parent.resolve()
        template_dir = project_root / "graphyflow" / "project_template"

    if not template_dir.exists():
        raise FileNotFoundError(f"Project template directory not found at: {template_dir}")

    # 2. 创建输出目录结构
    print(f"[1/6] Setting up Output Directory: '{output_dir}'")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    scripts_dir = output_dir / "scripts"
    host_script_dir = scripts_dir / "host"
    kernel_script_dir = scripts_dir / "kernel"
    xclbin_dir = output_dir / "xclbin"

    host_script_dir.mkdir(parents=True, exist_ok=True)
    kernel_script_dir.mkdir(parents=True, exist_ok=True)
    xclbin_dir.mkdir(exist_ok=True)

    # 3. 复制静态文件
    print(f"[2/6] Copying Static Files from Template: '{template_dir}'")
    static_files_to_ignore = ["Makefile", "run.sh", "system.cfg", "*.template"]
    shutil.copytree(
        template_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*static_files_to_ignore)
    )

    # 4. 动态生成需要模板化的脚本文件
    print("[3/6] Generating Templated Scripts...")
    replacements = {"{{KERNEL_NAME}}": kernel_name, "{{EXECUTABLE_NAME}}": executable_name}
    _copy_and_template(template_dir / "Makefile", output_dir / "Makefile", replacements)
    _copy_and_template(template_dir / "run.sh", output_dir / "run.sh", replacements)
    (output_dir / "run.sh").chmod(0o755)
    _copy_and_template(template_dir / "system.cfg", output_dir / "system.cfg", replacements)

    # 5. 实例化后端并生成所有动态代码
    print("[4/6] Generating Dynamic Source Code via BackendManager...")
    bkd_mng = BackendManager()
    kernel_h, kernel_cpp = bkd_mng.generate_backend(comp_col, global_graph, kernel_name)
    common_h = bkd_mng.generate_common_header(kernel_name)
    host_h, host_cpp = bkd_mng.generate_host_codes(kernel_name, template_dir / "scripts" / "host")

    # 6. 部署所有动态生成的文件
    print(f"[5/6] Deploying Generated Files to '{output_dir}'")
    with open(kernel_script_dir / f"{kernel_name}.h", "w") as f:
        f.write(kernel_h)
    with open(kernel_script_dir / f"{kernel_name}.cpp", "w") as f:
        f.write(kernel_cpp)
    with open(host_script_dir / "common.h", "w") as f:
        f.write(common_h)
    with open(host_script_dir / "generated_host.h", "w") as f:
        f.write(host_h)
    with open(host_script_dir / "generated_host.cpp", "w") as f:
        f.write(host_cpp)

    print("[6/6] Project Generation Complete!")

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/Makefile`:

```
# This file is auto-generated by the GraphyFlow backend.
KERNEL_NAME := {{KERNEL_NAME}}
EXECUTABLE := {{EXECUTABLE_NAME}}

TARGET ?= sw_emu
DEVICE := /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm

.PHONY: all clean cleanall exe check

include scripts/main.mk

# The 'check' target now runs the script with the TARGET variable
check: all
	./run.sh $(TARGET)

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/gen_random_graph.py`:

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random
import argparse


def generate_random_connected_graph(num_nodes, num_edges, has_weight, undirected=1):
    """
    生成一个随机的有向连通图。

    Args:
        num_nodes (int): 图中的节点数。
        num_edges (int): 图中的边数。
        has_weight (bool): 边是否应该有权重。

    Returns:
        list: 一个表示图的元组列表。
              如果无权重，格式为 [(u, v), ...]。
              如果有权重，格式为 [(u, v, w), ...]。
    """
    if num_nodes <= 0:
        return []

    # 节点编号从 0 到 num_nodes - 1
    nodes = list(range(num_nodes))
    edges = set()

    # --- 步骤 1: 确保图是连通的 ---
    # 我们通过创建一个随机生成树来保证弱连通性。
    # 从一个随机节点开始，然后逐渐将其他节点添加到树中。

    # 跟踪已在树中的节点和尚未在树中的节点
    in_tree = {random.choice(nodes)}
    not_in_tree = list(set(nodes) - in_tree)
    random.shuffle(not_in_tree)

    # 遍历所有尚未在树中的节点
    for node_to_add in not_in_tree:
        # 从已经在树中的节点中随机选择一个连接点
        connect_to = random.choice(list(in_tree))

        # 随机决定边的方向
        u, v = (node_to_add, connect_to) if random.random() < 0.5 else (connect_to, node_to_add)

        edges.add((u, v))
        in_tree.add(node_to_add)

    # --- 步骤 2: 添加剩余的边以达到指定的边数 ---
    # 我们现在已经有了 num_nodes - 1 条边，构成了一个连通的骨架。
    while len(edges) < num_edges:
        u = random.choice(nodes)
        v = random.choice(nodes)

        # 避免自环和重复的边
        if u != v and (u, v) not in edges:
            edges.add((u, v))

    # --- 步骤 3: 如果需要，为每条边分配随机权重 ---
    final_graph = []
    if has_weight:
        # 权重范围可以根据需要调整
        min_weight, max_weight = 1, 100
        for u, v in edges:
            weight = random.randint(min_weight, max_weight)
            final_graph.append((u, v, weight))
            if undirected:
                final_graph.append((v, u, weight))
    else:
        final_graph = list(edges)
        if undirected:
            final_graph = [(u, v) for u, v in edges] + [(v, u) for u, v in edges]

    # 最后打乱顺序，使输出看起来更随机
    random.shuffle(final_graph)

    return final_graph


def main():
    """主函数，用于解析命令行参数并打印生成的图。"""
    parser = argparse.ArgumentParser(
        description="生成一个随机的有向连通图。",
        epilog="例如: python gen_random_graph.py 10 15 --have_weight",
    )
    parser.add_argument("nodes", type=int, help="图中的节点数 (一个正整数)")
    parser.add_argument("edges", type=int, help="图中的边数")
    parser.add_argument("--have_weight", action="store_true", help="如果指定此标志，则为边生成随机权重")

    args = parser.parse_args()

    # --- 参数验证 ---
    num_nodes = args.nodes
    num_edges = args.edges

    if num_nodes <= 0:
        print(f"错误: 节点数必须是正数。", file=sys.stderr)
        sys.exit(1)

    min_required_edges = num_nodes - 1
    # 对于有向图，不考虑自环时，最大边数为 N * (N - 1)
    max_possible_edges = num_nodes * (num_nodes - 1)

    if num_edges < min_required_edges:
        print(f"错误: 对于 {num_nodes} 个节点的连通图, 至少需要 {min_required_edges} 条边。", file=sys.stderr)
        sys.exit(1)

    if num_edges > max_possible_edges:
        print(
            f"错误: 对于 {num_nodes} 个节点, 最多只能有 {max_possible_edges} 条边 (无自环)。", file=sys.stderr
        )
        sys.exit(1)

    # --- 生成并打印图 ---
    graph = generate_random_connected_graph(num_nodes, num_edges, args.have_weight)

    for edge_info in graph:
        # 使用 ' '.join() 可以优雅地处理元组中的所有元素
        print(" ".join(map(str, edge_info)))


if __name__ == "__main__":
    main()

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/run.sh`:

```sh
#!/bin/bash
# This file is auto-generated by the GraphyFlow backend.

# --- *** 关键修正：使其能够处理不同的目标 *** ---
TARGET=$1
EXECUTABLE="{{EXECUTABLE_NAME}}"
KERNEL="{{KERNEL_NAME}}"

# 默认目标为 sw_emu
if [ -z "$TARGET" ]; then
    TARGET="sw_emu"
fi

echo "--- Running for target: $TARGET ---"

# 1. 设置环境变量
source /home/feiyang/set_env.sh

if [ "$TARGET" = "sw_emu" ] || [ "$TARGET" = "hw_emu" ]; then
    export XCL_EMULATION_MODE=$TARGET
    echo "XCL_EMULATION_MODE set to: $XCL_EMULATION_MODE"
else
    # 对于 'hw'，取消设置 XCL_EMULATION_MODE
    unset XCL_EMULATION_MODE
    echo "Running on hardware, XCL_EMULATION_MODE is unset."
fi

# 确保 LD_PRELOAD 仍然生效
export LD_PRELOAD=/lib/x86_64-linux-gnu/libOpenCL.so.1

# 2. 动态构建 .xclbin 文件路径
XCLBIN_FILE="./xclbin/${KERNEL}.${TARGET}.xclbin"
if [ ! -f "$XCLBIN_FILE" ]; then
    echo "Error: XCLBIN file not found at '$XCLBIN_FILE'"
    echo "Please make sure the project is built for the target '$TARGET' by running 'make all TARGET=$TARGET'"
    exit 1
fi

# 3. 运行 host 程序
DATASET="./graph.txt"
./${EXECUTABLE} ${XCLBIN_FILE} $DATASET

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/clean.mk`:

```mk
cleanexe:
	-$(RMDIR) $(EXECUTABLE)

clean:
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) .Xil
	-$(RMDIR) *.zip
	-$(RMDIR) *.str
	-$(RMDIR) ./_x
	-$(RMDIR) ./membership.out
	-$(RMDIR) .run
	-$(RMDIR) makefile_gen
	-$(RMDIR) .ipcache

cleanall:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*,*hw*} 
	-$(RMDIR) sdaccel_* TempConfig system_estimate.xtxt *.rpt
	-$(RMDIR) src/*.ll _xocc_* .Xil dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) .Xil
	-$(RMDIR) *.zip
	-$(RMDIR) *.str
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) ./_x
	-$(RMDIR) ./membership.out
	-$(RMDIR) xclbin*
	-$(RMDIR) .run
	-$(RMDIR) makefile_gen
	-$(RMDIR) .ipcache
	-$(RMDIR) *.csv
	-$(RMDIR) *.protoinst

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/help.mk`:

```mk
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to generate the design for specified Target and Device."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) ""

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/fpga_executor.cpp`:

```cpp

#include "fpga_executor.h"
#include "generated_host.h"
#include <iostream>

#define KERNEL_NAME "graphyflow"

// ... (fpga_executor.cpp 的其余内容保持不变) ...
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec) {
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device,
                                      CL_QUEUE_PROFILING_ENABLE, &err));
    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    OCL_CHECK(err, cl::Program program(context, {device}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel kernel(program, KERNEL_NAME, &err));

    AlgorithmHost algo_host(context, kernel, q);
    algo_host.setup_buffers(graph, start_node);
    total_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;
    std::cout << "\nStarting FPGA execution..." << std::endl;

    // 对于流式内核, 这个循环只会执行一次 (因为 get_stop_flag 返回 1)
    // This loop now performs Bellman-Ford iterations until convergence or
    // max_iterations
    for (iter = 0; iter < max_iterations; ++iter) {
        algo_host.transfer_data_to_fpga();
        cl::Event event;
        algo_host.execute_kernel_iteration(event);
        event.wait();
        algo_host.transfer_data_from_fpga();

        unsigned long start = 0, end = 0;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        double iteration_time_ns = end - start;
        total_kernel_time_sec += iteration_time_ns * 1.0e-9;
        double mteps =
            (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

        std::cout << "FPGA Iteration " << iter << ": "
                  << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                  << "Throughput = " << mteps << " MTEPS" << std::endl;

        if (algo_host.check_convergence_and_update()) {
            std::cout << "FPGA computation converged after " << iter + 1
                      << " iteration(s)." << std::endl;
            break;
        }
    }

    const std::vector<int> &final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;

    return final_results;
}

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/fpga_executor.h`:

```h
#ifndef __FPGA_EXECUTOR_H__
#define __FPGA_EXECUTOR_H__

#include "common.h"
#include <vector>

// 通用的 FPGA 执行函数。
// 它通过 AlgorithmHost 类来处理所有与具体算法相关的操作。
std::vector<int> run_fpga_kernel(const std::string &xclbin_path,
                                 const GraphCSR &graph, int start_node,
                                 double &total_kernel_time_sec);

#endif // __FPGA_EXECUTOR_H__

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/graph_loader.cpp`:

```cpp
#include "graph_loader.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// 边结构体，用于临时存储从文件中读取的边
struct Edge {
    int src, dest, weight;
};

// 主函数，从文件中加载图并转换为 CSR 格式
GraphCSR load_graph_from_file(const std::string &file_path) {
    // ---- 1. 根据文件扩展名判断图的格式 ----
    bool is_one_based = false; // 默认为 0-indexed
    char comment_char = '#';   // 默认注释符

    if (file_path.size() > 4 &&
        file_path.substr(file_path.size() - 4) == ".mtx") {
        is_one_based = true; // .mtx 文件通常是 1-indexed
        comment_char = '%';  // .mtx 文件使用 '%' 作为注释
        std::cout << "Detected .mtx format (1-based indexing)." << std::endl;
    } else {
        std::cout << "Detected .txt format (0-based indexing)." << std::endl;
    }

    // ---- 2. 打开文件并逐行解析 ----
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open graph file: " << file_path
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<Edge> edges;
    int max_vertex_id = -1;
    std::string line;
    long line_num = 0;

    while (std::getline(file, line)) {
        line_num++;
        // 跳过空行和注释行
        if (line.empty() || line[0] == comment_char) {
            continue;
        }

        std::istringstream iss(line);
        Edge edge;

        // 尝试读取 src, dest, weight
        if (iss >> edge.src >> edge.dest >> edge.weight) {
            // 成功读取三个值
        } else {
            // 如果失败，重置流并尝试只读取 src, dest
            iss.clear();
            iss.seekg(0);
            if (iss >> edge.src >> edge.dest) {
                edge.weight = 1; // 赋予默认权重 1
            } else {
                std::cerr << "Warning: Skipping malformed line " << line_num
                          << ": " << line << std::endl;
                continue;
            }
        }

        // 如果是 1-based 格式，转换为 0-based
        if (is_one_based) {
            edge.src--;
            edge.dest--;
        }

        // 检查顶点ID是否有效
        if (edge.src < 0 || edge.dest < 0) {
            std::cerr
                << "Warning: Skipping edge with negative vertex ID on line "
                << line_num << std::endl;
            continue;
        }

        edges.push_back(edge);
        max_vertex_id = std::max({max_vertex_id, edge.src, edge.dest});
    }
    file.close();

    // ---- 3. 将边列表转换为 CSR 格式 ----
    GraphCSR graph;
    if (max_vertex_id == -1) { // 如果文件为空或无效
        graph.num_vertices = 0;
        graph.num_edges = 0;
        std::cout << "Graph is empty." << std::endl;
        return graph;
    }

    graph.num_vertices = max_vertex_id + 1;
    graph.num_edges = edges.size();

    std::cout << "Graph loaded: " << graph.num_vertices << " vertices, "
              << graph.num_edges << " edges." << std::endl;

    // 为了进行 CSR 转换，按源顶点 ID 对边进行排序
    std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) {
        if (a.src != b.src) {
            return a.src < b.src;
        }
        return a.dest < b.dest;
    });

    // 分配 CSR 数组内存
    graph.offsets.resize(graph.num_vertices + 1);
    graph.columns.resize(graph.num_edges);
    graph.weights.resize(graph.num_edges);

    // 填充 columns 和 weights 数组，并计算每个顶点的出度
    std::vector<int> out_degree(graph.num_vertices, 0);
    for (int i = 0; i < graph.num_edges; ++i) {
        graph.columns[i] = edges[i].dest;
        graph.weights[i] = edges[i].weight;
        out_degree[edges[i].src]++;
    }

    // 通过出度的前缀和计算 offsets 数组
    graph.offsets[0] = 0;
    for (int i = 0; i < graph.num_vertices; ++i) {
        graph.offsets[i + 1] = graph.offsets[i] + out_degree[i];
    }

    return graph;
}
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/graph_loader.h`:

```h
#ifndef __GRAPH_LOADER_H__
#define __GRAPH_LOADER_H__

#include "common.h"

// Loads a graph from a text file (edge list format: src dst weight)
// and converts it into a GraphCSR object.
GraphCSR load_graph_from_file(const std::string &file_path);

#endif // __GRAPH_LOADER_H__
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/host.cpp`:

```cpp
#include "common.h"
#include "fpga_executor.h" // <-- 修改: 包含新的执行器
#include "graph_loader.h"
#include "host_verifier.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <xclbin_file> <graph_data_file>"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbin_file = argv[1];
    std::string graph_file = argv[2];
    int start_node = 0;

    // 1. 加载图数据 (不变)
    std::cout << "--- Step 1: Loading Graph Data ---" << std::endl;
    GraphCSR graph = load_graph_from_file(graph_file);
    if (graph.num_vertices == 0) {
        return EXIT_FAILURE;
    }

    // 2. 在 FPGA 上运行 (调用新的通用执行器)
    std::cout << "\n--- Step 2: Running on FPGA ---" << std::endl;
    double total_kernel_time_sec = 0;
    std::vector<int> fpga_distances =
        run_fpga_kernel(xclbin_file, graph, start_node, total_kernel_time_sec);

    // 3. 在 Host CPU 上验证 (不变, 按你的要求保留)
    std::cout << "\n--- Step 3: Verifying on Host CPU ---" << std::endl;
    std::vector<int> host_distances = verify_on_host(graph, start_node);

    // 4. 比较结果 (不变)
    std::cout << "\n--- Step 4: Comparing Results ---" << std::endl;
    int error_count = 0;
    for (int i = 0; i < graph.num_vertices; ++i) {
        if (fpga_distances[i] != host_distances[i]) {
            if (error_count < 10) {
                std::cout << "Mismatch at vertex " << i << ": "
                          << "FPGA_Result = " << fpga_distances[i] << ", "
                          << "Host_Result = " << host_distances[i] << std::endl;
            }
            error_count++;
        }
    }

    // 5. 最终报告 (不变)
    std::cout << "\n--- Final Report ---" << std::endl;
    if (error_count == 0) {
        std::cout << "SUCCESS: Results match!" << std::endl;
    } else {
        std::cout << "FAILURE: Found " << error_count << " mismatches."
                  << std::endl;
    }

    std::cout << "Total FPGA Kernel Execution Time: "
              << total_kernel_time_sec * 1000.0 << " ms" << std::endl;
    std::cout << "Total MTEPS (Edges / Total Time): "
              << ((double)graph.num_edges * (double)graph.num_vertices) /
                     total_kernel_time_sec / 1.0e6
              << " MTEPS" << std::endl;

    return (error_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/host.mk`:

```mk
# Makefile for the Host Application
CXX := g++

# Executable name (由顶层 Makefile 传入)
# EXECUTABLE := graphyflow_host

# Source files
HOST_SRCS := scripts/host/host.cpp \
             scripts/host/graph_loader.cpp \
             scripts/host/fpga_executor.cpp \
             scripts/host/generated_host.cpp \
             scripts/host/host_verifier.cpp \
             scripts/host/host_bellman_ford.cpp \
             scripts/host/xcl2.cpp

# Include directories
# 新增: 将生成的 kernel 目录也加入 include 路径, 内核头文件名与内核名相同
CXXFLAGS := -Iscripts/host -Iscripts/kernel
CXXFLAGS += -I$(XILINX_XRT)/include
CXXFLAGS += -I$(XILINX_VITIS)/include
CXXFLAGS += -I$(XILINX_HLS)/include

# Compiler flags
CXXFLAGS += -std=c++17 -Wall

# Linker flags
LDFLAGS := -L$(XILINX_XRT)/lib
LDFLAGS += -lOpenCL -lxrt_coreutil -lstdc++ -lrt -pthread -Wl,--export-dynamic

# Build rule for the executable
$(EXECUTABLE): $(HOST_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_SRCS) -o $(EXECUTABLE) $(LDFLAGS)
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/host_bellman_ford.cpp`:

```cpp
#include "host_bellman_ford.h"

bool host_bellman_ford_iteration(const GraphCSR &graph,
                                 std::vector<int> &distances) {
    bool changed = false;

    // --- USER MODIFIABLE SECTION: Host Computation Logic ---
    // For each vertex, relax all outgoing edges
    for (int u = 0; u < graph.num_vertices; ++u) {
        if (distances[u] != INFINITY_DIST) {
            for (int i = graph.offsets[u]; i < graph.offsets[u + 1]; ++i) {
                int v = graph.columns[i];
                int weight = graph.weights[i];

                // Relaxation step
                if (distances[u] + weight < distances[v]) {
                    distances[v] = distances[u] + weight;
                    changed = true;
                }
            }
        }
    }
    // --- END USER MODIFIABLE SECTION ---

    return changed;
}
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/host_bellman_ford.h`:

```h
#ifndef __HOST_BELLMAN_FORD_H__
#define __HOST_BELLMAN_FORD_H__

#include "common.h"

// This function performs a single iteration of the Bellman-Ford algorithm on
// the host CPU. It returns true if any distance value was updated, false
// otherwise.
bool host_bellman_ford_iteration(const GraphCSR &graph,
                                 std::vector<int> &distances);

#endif // __HOST_BELLMAN_FORD_H__
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/host_verifier.cpp`:

```cpp
#include "host_verifier.h"
#include "host_bellman_ford.h"
#include <iostream>

std::vector<int> verify_on_host(const GraphCSR &graph, int start_node) {
    std::vector<int> distances(graph.num_vertices, INFINITY_DIST);
    distances[start_node] = 0;

    int max_iterations = graph.num_vertices;
    int iter = 0;
    bool changed = true;

    std::cout << "\nStarting Host verification..." << std::endl;

    while (changed && iter < max_iterations) {
        // Run one iteration of the algorithm
        changed = host_bellman_ford_iteration(graph, distances);
        iter++;
    }

    std::cout << "Host computation converged after " << iter << " iterations."
              << std::endl;

    // Check for negative weight cycles (optional but good practice)
    if (iter == max_iterations &&
        host_bellman_ford_iteration(graph, distances)) {
        std::cout << "Warning: Negative weight cycle detected by host verifier."
                  << std::endl;
    }

    return distances;
}
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/host_verifier.h`:

```h
#ifndef __HOST_VERIFIER_H__
#define __HOST_VERIFIER_H__

#include "common.h"

// Main function to run the Bellman-Ford algorithm on the host CPU for
// verification.
std::vector<int> verify_on_host(const GraphCSR &graph, int start_node);

#endif // __HOST_VERIFIER_H__
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/xcl2.cpp`:

```cpp
/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "xcl2.h"
#include <climits>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#if defined(_WINDOWS)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace xcl {
std::vector<cl::Device> get_devices(const std::string &vendor_name) {
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++) {
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName =
                           platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (!(platformName.compare(vendor_name))) {
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        std::cout << "Found the following platforms : " << std::endl;
        for (size_t j = 0; j < platforms.size(); j++) {
            platform = platforms[j];
            OCL_CHECK(err, std::string platformName =
                               platform.getInfo<CL_PLATFORM_NAME>(&err));
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
        }
        exit(EXIT_FAILURE);
    }
    // Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    OCL_CHECK(err,
              err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}

std::vector<cl::Device> get_xil_devices() { return get_devices("Xilinx"); }

cl::Device find_device_bdf(const std::vector<cl::Device> &devices,
                           const std::string &bdf) {
    char device_bdf[20];
    cl_int err;
    cl::Device device;
    int cnt = 0;
    for (uint32_t i = 0; i < devices.size(); i++) {
        OCL_CHECK(err,
                  err = devices[i].getInfo(CL_DEVICE_PCIE_BDF, &device_bdf));
        if (bdf == device_bdf) {
            device = devices[i];
            cnt++;
            break;
        }
    }
    if (cnt == 0) {
        std::cout << "Invalid device bdf. Please check and provide valid bdf\n";
        exit(EXIT_FAILURE);
    }
    return device;
}
cl_device_id find_device_bdf_c(cl_device_id *devices, const std::string &bdf,
                               cl_uint device_count) {
    char device_bdf[20];
    cl_int err;
    cl_device_id device;
    int cnt = 0;
    for (uint32_t i = 0; i < device_count; i++) {
        err = clGetDeviceInfo(devices[i], CL_DEVICE_PCIE_BDF,
                              sizeof(device_bdf), device_bdf, 0);
        if (err != CL_SUCCESS) {
            std::cout << "Unable to extract the device BDF details\n";
            exit(EXIT_FAILURE);
        }
        if (bdf == device_bdf) {
            device = devices[i];
            cnt++;
            break;
        }
    }
    if (cnt == 0) {
        std::cout << "Invalid device bdf. Please check and provide valid bdf\n";
        exit(EXIT_FAILURE);
    }
    return device;
}
std::vector<unsigned char>
read_binary_file(const std::string &xclbin_file_name) {
    std::cout << "INFO: Reading " << xclbin_file_name << std::endl;
    FILE *fp;
    if ((fp = fopen(xclbin_file_name.c_str(), "r")) == nullptr) {
        printf("ERROR: %s xclbin not available please build\n",
               xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    // Loading XCL Bin into char buffer
    std::cout << "Loading: '" << xclbin_file_name.c_str() << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    auto nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    std::vector<unsigned char> buf;
    buf.resize(nb);
    bin_file.read(reinterpret_cast<char *>(buf.data()), nb);
    return buf;
}

bool is_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if (xcl_mode != nullptr) {
        ret = true;
    }
    return ret;
}

bool is_hw_emulation() {
    bool ret = false;
    char *xcl_mode = getenv("XCL_EMULATION_MODE");
    if ((xcl_mode != nullptr) && !strcmp(xcl_mode, "hw_emu")) {
        ret = true;
    }
    return ret;
}
double round_off(double n) {
    double d = n * 100.0;
    int i = d + 0.5;
    d = i / 100.0;
    return d;
}

std::string convert_size(size_t size) {
    static const char *SIZES[] = {"B", "KB", "MB", "GB"};
    uint32_t div = 0;
    size_t rem = 0;

    while (size >= 1024 && div < (sizeof SIZES / sizeof *SIZES)) {
        rem = (size % 1024);
        div++;
        size /= 1024;
    }

    double size_d = (float)size + (float)rem / 1024.0;
    double size_val = round_off(size_d);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << size_val;
    std::string size_str = stream.str();
    std::string result = size_str + " " + SIZES[div];
    return result;
}

bool is_xpr_device(const char *device_name) {
    const char *output = strstr(device_name, "xpr");

    if (output == nullptr) {
        return false;
    } else {
        return true;
    }
}
}; // namespace xcl

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/host/xcl2.h`:

```h
/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

// OCL_CHECK doesn't work if call has templatized function call
#define OCL_CHECK(error, call)                                                 \
    call;                                                                      \
    if (error != CL_SUCCESS) {                                                 \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, \
               __LINE__, error);                                               \
        exit(EXIT_FAILURE);                                                    \
    }

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>
#include <fstream>
#include <iostream>
// When creating a buffer with user pointer (CL_MEM_USE_HOST_PTR), under the
// hood
// User ptr is used if and only if it is properly aligned (page aligned). When
// not
// aligned, runtime has no choice but to create its own host side buffer that
// backs
// user ptr. This in turn implies that all operations that move data to and from
// device incur an extra memcpy to move data to/from runtime's own host buffer
// from/to user pointer. So it is recommended to use this allocator if user wish
// to
// Create Buffer/Memory Object with CL_MEM_USE_HOST_PTR to align user buffer to
// the
// page boundary. It will ensure that user buffer will be used when user create
// Buffer/Mem Object with CL_MEM_USE_HOST_PTR.
template <typename T> struct aligned_allocator {
    using value_type = T;

    aligned_allocator() {}

    aligned_allocator(const aligned_allocator &) {}

    template <typename U> aligned_allocator(const aligned_allocator<U> &) {}

    T *allocate(std::size_t num) {
        void *ptr = nullptr;

#if defined(_WINDOWS)
        {
            ptr = _aligned_malloc(num * sizeof(T), 4096);
            if (ptr == nullptr) {
                std::cout << "Failed to allocate memory" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
#else
        {
            if (posix_memalign(&ptr, 4096, num * sizeof(T)))
                throw std::bad_alloc();
        }
#endif
        return reinterpret_cast<T *>(ptr);
    }
    void deallocate(T *p, std::size_t num) {
#if defined(_WINDOWS)
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

namespace xcl {
std::vector<cl::Device> get_xil_devices();
std::vector<cl::Device> get_devices(const std::string &vendor_name);
cl::Device find_device_bdf(const std::vector<cl::Device> &devices,
                           const std::string &bdf);
cl_device_id find_device_bdf_c(cl_device_id *devices, const std::string &bdf,
                               cl_uint dev_count);
std::string convert_size(size_t size);
std::vector<unsigned char>
read_binary_file(const std::string &xclbin_file_name);
bool is_emulation();
bool is_hw_emulation();
bool is_xpr_device(const char *device_name);
class P2P {
  public:
    static decltype(&xclGetMemObjectFd) getMemObjectFd;
    static decltype(&xclGetMemObjectFromFd) getMemObjectFromFd;
    static void init(const cl_platform_id &platform) {
        void *bar = clGetExtensionFunctionAddressForPlatform(
            platform, "xclGetMemObjectFd");
        getMemObjectFd = (decltype(&xclGetMemObjectFd))bar;
        bar = clGetExtensionFunctionAddressForPlatform(platform,
                                                       "xclGetMemObjectFromFd");
        getMemObjectFromFd = (decltype(&xclGetMemObjectFromFd))bar;
    }
};
class Ext {
  public:
    static decltype(&xclGetComputeUnitInfo) getComputeUnitInfo;
    static void init(const cl_platform_id &platform) {
        void *bar = clGetExtensionFunctionAddressForPlatform(
            platform, "xclGetComputeUnitInfo");
        getComputeUnitInfo = (decltype(&xclGetComputeUnitInfo))bar;
    }
};
} // namespace xcl

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/kernel/kernel.mk`:

```mk

# Makefile for the Vitis Kernel
VPP := v++
# KERNEL_NAME is passed from the top Makefile
KERNEL_SRC := scripts/kernel/$(KERNEL_NAME).cpp
XCLBIN_DIR := ./xclbin
XCLBIN_FILE := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xclbin
KERNEL_XO := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xo
EMCONFIG_FILE := ./emconfig.json
CLFLAGS += --kernel $(KERNEL_NAME)
CLFLAGS += -Iscripts/kernel
CLFLAGS += -Iscripts/host
CLFLAGS += -I$(XILINX_XRT)/include
CLFLAGS += -I$(XILINX_VITIS)/include
LDFLAGS_VPP += --config ./system.cfg
LDFLAGS_VPP += -Iscripts/kernel
LDFLAGS_VPP += -Iscripts/host
LDFLAGS_VPP += -I$(XILINX_XRT)/include
LDFLAGS_VPP += -I$(XILINX_VITIS)/include

$(KERNEL_XO): $(KERNEL_SRC)
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) $(CLFLAGS) -o $@ $<

$(XCLBIN_FILE): $(KERNEL_XO)
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) $(LDFLAGS_VPP) -o $@ $<

emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/main.mk`:

```mk
SHELL           := /bin/bash

COMMON_REPO     = ./
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))
SCRIPTS_PATH    = ./scripts

# Remove unused targets from .PHONY
.PHONY: all clean cleanall exe emconfig

include $(SCRIPTS_PATH)/help.mk
include $(SCRIPTS_PATH)/utils.mk

include global_para.mk

# Use SCRIPTS_PATH for consistency
include $(SCRIPTS_PATH)/host/host.mk 
# Remove non-existent makefiles
# include autogen/autogen.mk
# include acc_template/acc.mk

# Include our new kernel makefile
include $(SCRIPTS_PATH)/kernel/kernel.mk

# This include seems to be for Vitis 1.0 examples, not needed here
# include $(SCRIPTS_PATH)/bitstream.mk
include $(SCRIPTS_PATH)/clean.mk

# Update the 'all' rule to depend on the .xclbin file, the host executable, and emconfig
all: $(XCLBIN_FILE) $(EXECUTABLE) emconfig

exe: $(EXECUTABLE)
```

`CCFSys2025_GraphyFlow/graphyflow/project_template/scripts/utils.mk`:

```mk
#+-------------------------------------------------------------------------------
# The following parameters are assigned with default values. These parameters can
# be overridden through the make command line
#+-------------------------------------------------------------------------------

PROFILE := no

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all
endif

DEBUG := no

#Generates debug summary report
ifeq ($(DEBUG), yes)
CLFLAGS += --dk protocol:all:all:all
endif

#Generates debug summary report
ifeq ($(DEBUG), yes)
LDCLFLAGS += --dk list_ports
endif

#Checks for XILINX_VITIS
ifndef XILINX_VITIS
$(error XILINX_VITIS variable is not set, please set correctly and rerun)
endif

#Checks for XILINX_XRT
check-xrt:
ifndef XILINX_XRT
	$(error XILINX_XRT variable is not set, please set correctly and rerun)
endif

check-devices:
ifndef DEVICE
	$(error DEVICE not set. Please set the DEVICE properly and rerun. Run "make help" for more details.)
endif

check-aws_repo:
ifndef SDACCEL_DIR
	$(error SDACCEL_DIR not set. Please set it properly and rerun. Run "make help" for more details.)
endif

#   sanitize_dsa - create a filesystem friendly name from dsa name
#   $(1) - name of dsa
COLON=:
PERIOD=.
UNDERSCORE=_
sanitize_dsa = $(strip $(subst $(PERIOD),$(UNDERSCORE),$(subst $(COLON),$(UNDERSCORE),$(1))))

device2dsa = $(if $(filter $(suffix $(1)),.xpfm),$(shell $(COMMON_REPO)/utility/parsexpmf.py $(1) dsa 2>/dev/null),$(1))
device2sandsa = $(call sanitize_dsa,$(call device2dsa,$(1)))
device2dep = $(if $(filter $(suffix $(1)),.xpfm),$(dir $(1))/$(shell $(COMMON_REPO)/utility/parsexpmf.py $(1) hw 2>/dev/null) $(1),)

# Cleaning stuff
RM = rm -f
RMDIR = rm -rf

ECHO:= @echo

```

`CCFSys2025_GraphyFlow/graphyflow/project_template/system.cfg`:

```cfg
# This file is auto-generated by the GraphyFlow backend.
[connectivity]
nk={{KERNEL_NAME}}:1:{{KERNEL_NAME}}_1

```

`CCFSys2025_GraphyFlow/graphyflow/simulate.py`:

```py
import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from typing import Dict, List, Tuple, Any, Callable, Optional, Set
import collections


class UncertainArray:
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"UncertainArray({self.value})"


def _simulate_component(
    comp: dfir.Component,
    inputs: Dict[str, Any],
    node_props: Dict[int, Dict[str, Any]],
    edge_props: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Simulates a single DFIR component based on its type and inputs.
    Handles parallel (array) operations where applicable.
    """
    # Remove TODO for Array<> Inputs as we are addressing it now
    support_components = (
        dfir.BinOpComponent,
        dfir.UnaryOpComponent,
        dfir.ConstantComponent,
        dfir.CopyComponent,
        dfir.GatherComponent,
        dfir.ScatterComponent,
        dfir.ConditionalComponent,
        dfir.CollectComponent,
        dfir.PlaceholderComponent,
        dfir.ReduceComponent,  # Add ReduceComponent here as _simulate_component handles it
    )
    # Check if comp is one of the base supported types or ReduceComponent
    assert isinstance(
        comp, support_components
    ), f"Component type {type(comp).__name__} is not directly supported by _simulate_component base logic or ReduceComponent logic."

    # --- Input Port Check (Moved earlier for clarity) ---
    # For ReduceComponent, only 'i_0' might be directly available when called *by* run_flow's loop.
    # Other components expect all their inputs.
    if not isinstance(comp, dfir.ReduceComponent):
        assert all(
            p.name in inputs.keys() for p in comp.in_ports
        ), f"Component {comp.uuid} ({type(comp).__name__}) is missing inputs. Got: {list(inputs.keys())}, Expected: {[p.name for p in comp.in_ports]}"

    # --- Component Simulation Logic ---
    if isinstance(comp, dfir.PlaceholderComponent):
        # Handles scalar or array pass-through
        return {"o_0": inputs["i_0"]}

    if isinstance(comp, dfir.ConstantComponent):
        # Handles scalar or array constant
        return {"o_0": comp.value if not comp.parallel else UncertainArray(comp.value)}

    if isinstance(comp, dfir.CopyComponent):
        # Handles scalar or array copy (by reference for arrays)
        return {"o_0": inputs["i_0"], "o_1": inputs["i_0"]}

    if isinstance(comp, dfir.GatherComponent):
        # Input ports are i_0, i_1, ...
        # Output port is o_0
        input_values = [inputs[p.name] for p in comp.in_ports]

        if comp.parallel:
            # Ensure all input lists have the same length
            first_len = len(input_values[0])
            assert all(
                len(val) == first_len for val in input_values
            ), "Parallel GatherComponent requires all input arrays to have the same length"
            # Zip the lists element-wise: [[a1, b1], [a2, b2], ...] -> [(a1, a2), (b1, b2)]
            output_list = list(zip(*input_values, strict=True))  # Result is Array<Tuple<...>>
            return {"o_0": output_list}
        else:
            # Scalar inputs: gather into a single tuple
            output_tuple = tuple(input_values)  # Result is Tuple<...>
            return {"o_0": output_tuple}

    if isinstance(comp, dfir.ScatterComponent):
        # Input port is i_0
        # Output ports are o_0, o_1, ...
        input_val = inputs["i_0"]

        if comp.parallel:
            # Input is Array<Tuple<T...>>
            # Each element of input_val is a tuple to be scattered
            num_outputs = len(comp.out_ports)
            # Check if tuples have the correct size
            if input_val:  # Check if list is not empty
                assert all(
                    isinstance(item, (list, tuple)) and len(item) == num_outputs for item in input_val
                ), f"Parallel ScatterComponent input elements must be tuples/lists of length {num_outputs}"

            # Unzip the list of tuples: [(a1, b1), (a2, b2), ...] -> ([a1, a2], [b1, b2])
            scattered_outputs = list(zip(*input_val))

            return {p.name: list(scattered_outputs[i]) for i, p in enumerate(comp.out_ports)}
        else:
            # Input is Tuple<T...>
            assert isinstance(
                input_val, (list, tuple)
            ), "Non-parallel ScatterComponent expects tuple/list input"
            assert len(input_val) == len(
                comp.out_ports
            ), f"Non-parallel ScatterComponent input length mismatch: expected {len(comp.out_ports)}, got {len(input_val)}"
            # Scatter elements of the tuple to corresponding output ports
            return {p.name: input_val[i] for i, p in enumerate(comp.out_ports)}

    if isinstance(comp, dfir.BinOpComponent):
        op_func = {  # Keep original scalar functions
            dfir.BinOp.ADD: lambda x, y: x + y,
            dfir.BinOp.SUB: lambda x, y: x - y,
            dfir.BinOp.MUL: lambda x, y: x * y,
            dfir.BinOp.DIV: lambda x, y: x / y,
            dfir.BinOp.AND: lambda x, y: x & y,
            dfir.BinOp.OR: lambda x, y: x | y,
            dfir.BinOp.EQ: lambda x, y: x == y,
            dfir.BinOp.NE: lambda x, y: x != y,
            dfir.BinOp.LT: lambda x, y: x < y,
            dfir.BinOp.GT: lambda x, y: x > y,
            dfir.BinOp.LE: lambda x, y: x <= y,
            dfir.BinOp.GE: lambda x, y: x >= y,
            dfir.BinOp.MIN: lambda x, y: min(x, y),
            dfir.BinOp.MAX: lambda x, y: max(x, y),
        }[comp.op]
        in1 = inputs["i_0"]
        in2 = inputs["i_1"]

        if isinstance(in1, UncertainArray):
            assert not isinstance(in2, UncertainArray)
            in1 = [in1.value] * len(in2)
        if isinstance(in2, UncertainArray):
            assert not isinstance(in1, UncertainArray)
            in2 = [in2.value] * len(in1)

        # Assume comp has a 'parallel' attribute based on dfir definition
        if getattr(comp, "parallel", False):  # Check if parallel attribute exists and is True
            assert isinstance(in1, list) and isinstance(
                in2, list
            ), f"Parallel BinOp requires list inputs, got {in1=}, {in2=}"
            # Handle potential broadcast if lengths differ? For now, assume same length.
            assert len(in1) == len(in2), "Parallel BinOp requires inputs of same length"
            result = [op_func(x, y) for x, y in zip(in1, in2)]
        else:
            # Scalar operation
            result = op_func(in1, in2)
        return {"o_0": result}

    if isinstance(comp, dfir.UnaryOpComponent):
        input_val = inputs["i_0"]
        input_port_type = comp.in_ports[0].data_type

        op_func = {  # Keep original scalar functions
            dfir.UnaryOp.NOT: lambda x: not x,
            dfir.UnaryOp.NEG: lambda x: -x,
            dfir.UnaryOp.CAST_BOOL: lambda x: bool(x),
            dfir.UnaryOp.CAST_INT: lambda x: int(x),
            dfir.UnaryOp.CAST_FLOAT: lambda x: float(x),
            dfir.UnaryOp.SELECT: lambda x: x[comp.select_index],  # Array handling?
            dfir.UnaryOp.GET_LENGTH: lambda x: len(x),  # Array handling?
            dfir.UnaryOp.GET_ATTR: lambda x: (  # Scalar Node/Edge ID expected here
                node_props[x][comp.select_index]
                if isinstance(input_port_type.type_, dfir.SpecialType)
                and input_port_type.type_.type_name == "node"
                else (
                    edge_props[x][comp.select_index]
                    if isinstance(input_port_type.type_, dfir.SpecialType)
                    and input_port_type.type_.type_name == "edge"
                    else (_ for _ in ()).throw(
                        AssertionError(f"GET_ATTR expects input type 'node' or 'edge', got {input_port_type}")
                    )
                )
            ),
        }[comp.op]

        if isinstance(input_val, UncertainArray):
            assert comp.parallel, "UncertainArray input is only supported for parallel UnaryOp"
            result_val = op_func(input_val.value)
            return {"o_0": UncertainArray(result_val)}

        # Assume comp has a 'parallel' attribute
        if getattr(comp, "parallel", False):
            assert isinstance(input_val, list), "Parallel UnaryOp requires list input"
            result = [op_func(item) for item in input_val]
        else:
            result = op_func(input_val)
        return {"o_0": result}

    if isinstance(comp, dfir.ConditionalComponent):
        # This already handles the 'parallel' case correctly. No changes needed.
        data_in = inputs["i_data"]
        cond_in = inputs["i_cond"]
        if comp.parallel:
            assert isinstance(data_in, list), "Parallel ConditionalComponent expects 'i_data' to be a list"
            assert isinstance(cond_in, list), "Parallel ConditionalComponent expects 'i_cond' to be a list"
            assert len(data_in) == len(
                cond_in
            ), "Parallel ConditionalComponent requires 'i_data' and 'i_cond' lists to have the same length"
            output_list = [
                data_item if bool(cond_item) else None for data_item, cond_item in zip(data_in, cond_in)
            ]
            return {"o_0": output_list}
        else:
            return {"o_0": data_in if bool(cond_in) else None}

    if isinstance(comp, dfir.CollectComponent):
        # This operates on Array<Optional<T>> -> Array<T>. Handles arrays correctly. No changes needed.
        optional_data_in = inputs["i_0"]
        assert isinstance(optional_data_in, list), "CollectComponent expects 'i_0' to be a list"
        output_list = [item for item in optional_data_in if item is not None]
        return {"o_0": output_list}

    if isinstance(comp, dfir.ReduceComponent):
        input_array = inputs["i_0"]
        assert isinstance(input_array, list), "ReduceComponent expects 'i_0' to be a list"

        def find_subgraph_entry_port(start_port: dfir.Port) -> dfir.Port:
            assert start_port.port_type == dfir.PortType.OUT
            assert start_port.connected
            return start_port.connection

        def find_subgraph_exit_port(end_port: dfir.Port) -> dfir.Port:
            assert end_port.port_type == dfir.PortType.IN
            assert end_port.connected
            return end_port.connection

        key_entry = find_subgraph_entry_port(comp.get_port("o_reduce_key_in"))
        key_exit = find_subgraph_exit_port(comp.get_port("i_reduce_key_out"))
        transform_entry = find_subgraph_entry_port(comp.get_port("o_reduce_transform_in"))
        transform_exit = find_subgraph_exit_port(comp.get_port("i_reduce_transform_out"))
        accum_entry_0 = find_subgraph_entry_port(comp.get_port("o_reduce_unit_start_0"))
        accum_entry_1 = find_subgraph_entry_port(comp.get_port("o_reduce_unit_start_1"))
        reduce_exit = find_subgraph_exit_port(comp.get_port("i_reduce_unit_end"))
        groups = {}
        for element in input_array:
            key_result_dict = run_flow({key_entry: element}, [key_exit], node_props, edge_props)
            key = key_result_dict[key_exit]
            if key not in groups:
                groups[key] = []
            groups[key].append(element)
        final_results = []
        for key, elements_in_group in groups.items():
            if not elements_in_group:
                continue
            accumulated_value = None
            is_first = True
            for element in elements_in_group:
                if is_first:
                    reduction_result_dict = run_flow(
                        {transform_entry: element},
                        [transform_exit],
                        node_props,
                        edge_props,
                    )
                    accumulated_value = reduction_result_dict[transform_exit]
                    is_first = False
                else:
                    transform_result_dict = run_flow(
                        {transform_entry: element},
                        [transform_exit],
                        node_props,
                        edge_props,
                    )
                    reduction_result_dict = run_flow(
                        {
                            accum_entry_0: accumulated_value,
                            accum_entry_1: transform_result_dict[transform_exit],
                        },
                        [reduce_exit],
                        node_props,
                        edge_props,
                    )
                    accumulated_value = reduction_result_dict[reduce_exit]
            final_results.append(accumulated_value)
        return {"o_0": final_results}
    else:
        # Handle unsupported components
        raise NotImplementedError(
            f"Simulation logic not implemented in _simulate_component for: {type(comp).__name__}"
        )


def run_flow(
    from_ports_values: Dict[dfir.Port, Any],
    to_ports: List[dfir.Port],
    node_props: Dict[int, Dict[str, Any]],
    edge_props: Dict[int, Dict[str, Any]],
    run_no_input_components: List[dfir.Component] = [],
    data_for_io_comp=None,
) -> Dict[dfir.Port, Any]:
    """
    Runs a data flow simulation from a set of starting ports/values
    to a set of target output ports.

    Args:
        from_ports_values: Dict mapping starting input ports to their initial values.
                           Ports must be input ports.
        to_ports: List of target output ports. The simulation stops when all
                  these ports have computed values.
        node_props: Dictionary of node properties.
        edge_props: Dictionary of edge properties.

    Returns:
        A dictionary mapping the target output ports (from to_ports) to their
        computed values.
    """
    computed_values: Dict[dfir.Port, Any] = from_ports_values.copy()
    target_ports_set = set(to_ports)
    computed_target_ports: Set[dfir.Port] = set()
    final_results: Dict[dfir.Port, Any] = {}
    # Check if any target ports are already provided as inputs
    for port in from_ports_values:
        # Input ports shouldn't be target ports (which are outputs)
        assert port not in target_ports_set, f"Target port {port} cannot be an input port."

    ready_queue = collections.deque()
    processed_components = set()
    for start_port, _ in from_ports_values.items():
        assert (
            start_port.port_type == dfir.PortType.IN
        ), f"Port {start_port} in from_ports_values must be an input port."
        component = start_port.parent
        if component.uuid not in processed_components:
            assert all(
                in_port in computed_values for in_port in component.in_ports
            ), f"Component {component.uuid} has inputs not in {computed_values=}"
            ready_queue.append(component)
            processed_components.add(component.uuid)

    for component in list(set(run_no_input_components) - set(ready_queue)):
        assert len(component.in_ports) == 0, f"Component {component.uuid} has input ports"
        ready_queue.append(component)

    # --- Simulation Loop ---
    executed_component_ids = set()  # Track executed components to prevent cycles in simple cases

    while ready_queue and computed_target_ports != target_ports_set:
        component_to_run = ready_queue.popleft()

        # Prevent re-execution in this flow (basic cycle detection)
        assert (
            component_to_run.uuid not in executed_component_ids
        ), f"Component {component_to_run.uuid} already executed"
        executed_component_ids.add(component_to_run.uuid)

        # Gather inputs for this component
        current_inputs = {}
        if isinstance(component_to_run, dfir.ReduceComponent):
            # get the "i_0" port and assert it is in computed_values
            i_0_port = component_to_run.get_port("i_0")
            assert (
                i_0_port in computed_values
            ), f"Component {component_to_run.uuid} has input port 'i_0' not in {computed_values=}"
            current_inputs["i_0"] = computed_values[i_0_port]
        else:
            assert all(
                in_port in computed_values for in_port in component_to_run.in_ports
            ), f"Component {component_to_run.uuid} has inputs not in {computed_values=}"
            current_inputs = {in_port.name: computed_values[in_port] for in_port in component_to_run.in_ports}

        if isinstance(component_to_run, dfir.IOComponent):
            assert (
                data_for_io_comp is not None
            ), f"Data for IO component {component_to_run.uuid} is not provided"
            assert (
                component_to_run.io_type == dfir.IOComponent.IOType.INPUT
            ), f"IO component {component_to_run.uuid} is not an input component"
            outputs = {"o_0": data_for_io_comp[component_to_run.output_type.type_.type_name]}
        else:
            outputs = _simulate_component(component_to_run, current_inputs, node_props, edge_props)
        # print(f"Finished running component {component_to_run}, outputs: {outputs}")

        # --- Process outputs and update state ---
        for out_port_name, value in outputs.items():
            out_port = component_to_run.get_port(out_port_name)
            if out_port in target_ports_set:
                computed_target_ports.add(out_port)
                final_results[out_port] = value
                continue
            connected_in_port = out_port.connection
            assert connected_in_port is not None, f"Output port {out_port} is not connected to any input port"
            computed_values[connected_in_port] = value

            # Find downstream components and check readiness
            downstream_comp = connected_in_port.parent

            # Check if downstream component is now ready
            is_ready = all(in_port in computed_values for in_port in downstream_comp.in_ports)
            if (
                isinstance(downstream_comp, dfir.ReduceComponent)
                and downstream_comp.get_port("i_0") in computed_values
            ):
                is_ready = True
            if (
                is_ready
                and downstream_comp not in ready_queue
                and downstream_comp.uuid not in executed_component_ids
            ):
                ready_queue.append(downstream_comp)

    # --- Final Check and Return ---
    if computed_target_ports != target_ports_set:
        missing = target_ports_set - computed_target_ports
        raise RuntimeError(f"run_flow failed to compute all target ports. Missing: {missing}")

    return final_results


class DfirSimulator:
    def __init__(self, dfirs: dfir.ComponentCollection, g: GlobalGraph) -> None:
        self.dfirs = dfirs
        self.node_properties_schema = g.node_properties
        self.edge_properties_schema = g.edge_properties
        self.node_data = {}
        self.edge_data = {}

    def add_nodes(self, nodes: List[int], props: Dict[int, Dict[str, Any]]):
        assert len(set(nodes)) == len(nodes)
        self.node_data = {i: {} for i in nodes}
        for i in self.node_data.keys():
            assert i in props.keys()
            for prop_name in self.node_properties_schema:
                if prop_name == "id":
                    self.node_data[i][prop_name] = i
                    continue
                assert prop_name in props[i].keys(), f"Node {i} missing property '{prop_name}'"
                self.node_data[i][prop_name] = props[i][prop_name]

    def add_edges(self, edges: Dict[int, Tuple[int, int]], props: Dict[int, Dict[str, Any]]):
        assert len(set(edges.keys())) == len(edges.keys())
        self.edge_data = {i: {"src": v[0], "dst": v[1]} for i, v in edges.items()}
        for i in self.edge_data.keys():
            assert i in props.keys()
            for prop_name in self.edge_properties_schema:
                if prop_name in ["src", "dst"]:
                    continue
                assert prop_name in props[i].keys(), f"Edge {i} missing property '{prop_name}'"
                self.edge_data[i][prop_name] = props[i][prop_name]

    def run(self, initial_graph_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs the simulation for the entire dataflow graph.

        Args:
            initial_graph_inputs: A dictionary mapping the names of the graph's
                                  input ports to their initial values.

        Returns:
            A dictionary mapping the names of the graph's output ports to
            their final computed values.
        """
        graph_input_ports: List[dfir.Port] = self.dfirs.inputs
        graph_output_ports: List[dfir.Port] = self.dfirs.outputs

        from_ports_values: Dict[dfir.Port, Any] = {}
        provided_input_names = set(initial_graph_inputs.keys())
        used_input_names = set()

        for port in graph_input_ports:
            assert port.port_type == dfir.PortType.IN, f"Graph input port {port} is not an IN port."
            if port.name in initial_graph_inputs:
                from_ports_values[port] = initial_graph_inputs[port.name]
                used_input_names.add(port.name)
            else:
                raise ValueError(f"Initial value for graph input port '{port.name}' not provided.")

        unused_input_names = provided_input_names - used_input_names
        if unused_input_names:
            raise ValueError(
                f"Provided initial inputs do not correspond to graph input ports: {unused_input_names}"
            )

        to_ports: List[dfir.Port] = graph_output_ports
        for port in to_ports:
            assert (
                port.port_type == dfir.PortType.OUT
            ), f"Graph output port {port} expected to be an OUT port, but got {port.port_type}"

        no_input_components = []
        for component in self.dfirs.components:
            if len(component.in_ports) == 0:
                no_input_components.append(component)

        final_port_values: Dict[dfir.Port, Any] = run_flow(
            from_ports_values=from_ports_values,
            to_ports=to_ports,
            node_props=self.node_data,
            edge_props=self.edge_data,
            run_no_input_components=no_input_components,
            data_for_io_comp={
                "node": [node_id for node_id in self.node_data.keys()],
                "edge": [edge_id for edge_id in self.edge_data.keys()],
            },
        )

        final_results_by_name: Dict[str, Any] = {
            port.name: final_port_values[port] for port in to_ports if port in final_port_values
        }

        if len(final_results_by_name) != len(to_ports):
            computed_port_names = set(final_results_by_name.keys())
            requested_port_names = set(p.name for p in to_ports)
            missing_names = requested_port_names - computed_port_names
            print(
                f"Warning: Simulation finished, but some graph output ports were not computed by run_flow: {missing_names}"
            )

        return final_results_by_name


if __name__ == "__main__":
    print("Running Simulation Test...")

    g = GlobalGraph(properties={"node": {"weight": dfir.IntType()}, "edge": {}})
    nodes = g.add_graph_input("edge")
    src_dst_weight = nodes.map_(map_func=lambda edge: (edge.src.weight, edge.dst, edge))
    test = src_dst_weight.reduce_by(
        reduce_key=lambda data: data[1],
        reduce_transform=lambda data: data,
        reduce_method=lambda data1, data2: (data1[0] + data2[0], data1[1], data1[2]),
    )
    test2 = test.filter(filter_func=lambda data: data[0] > 10)
    # test2 = test.filter(filter_func=lambda data: 10 > data[0])
    # print(g)
    my_dfir = g.to_dfir()[0]
    # print(my_dfir)
    from graphyflow.visualize_ir import visualize_components

    dot = visualize_components(str(my_dfir))
    dot.render("component_graph", view=False, format="png")

    simulator = DfirSimulator(my_dfir, g)

    node_num = 20
    simulator.add_nodes(nodes=list(range(node_num)), props={i: {"weight": i} for i in range(node_num)})
    edges = []
    edges.extend([(i, i + 1) for i in range(node_num - 1)])
    edges.extend([(i, i + 2) for i in range(node_num - 2)])
    simulator.add_edges(
        edges={i * 10: edges[i] for i in range(len(edges))},
        props={i * 10: {} for i in range(len(edges))},
    )

    print(f"Running simulation")
    results = simulator.run({})
    print(f"Simulation results: {results}")

```

`CCFSys2025_GraphyFlow/graphyflow/visualize_ir.py`:

```py
import re
from graphviz import Digraph


def parse_components(text):
    # print(text)
    components = []
    port_to_component = {}

    # 首先提取组件列表部分
    components_match = re.search(r"components:\s*\[(.*?)\],\s*inputs:", text, re.DOTALL)
    if not components_match:
        return [], {}, [], []

    components_text = components_match.group(1)

    # 基于括号匹配来拆分组件
    def split_components_by_brackets(text):
        components = []
        current = ""
        bracket_level = 0
        in_component = False
        comp_st = 0

        for i, char in enumerate(text):
            if not in_component and "Component(" in text[max(0, i - 15) : i + 1]:
                in_component = True
                comp_st = i - 1
                while comp_st >= 0 and (text[comp_st] == "_" or text[comp_st].isalpha()):
                    comp_st -= 1
                comp_st += 1

            if in_component:
                if char == "(":
                    bracket_level += 1
                elif char == ")":
                    bracket_level -= 1
                    if bracket_level == 0:
                        components.append(text[comp_st : i + 1])
                        in_component = False

        return components

    component_texts = split_components_by_brackets(components_text)
    # print(("="*100 + "\n").join(component_texts))
    component_matches = [
        (re.match(r"(\w+)Component", comp).group(1), comp)
        for comp in component_texts
        if re.match(r"(\w+)Component", comp)
    ]

    for idx, (comp_type, comp_content) in enumerate(component_matches):
        # print(comp_type, comp_content)
        comp_id = f"comp_{idx}"

        component = {
            "id": comp_id,
            "type": comp_type + "Component",
            "label": comp_type,
            "ports": {},
            "connections": [],
        }

        # Extract value for ConstantComponent
        if comp_type == "Constant":
            value_match = re.search(r"value: (\d+)", comp_content)
            if value_match:
                component["label"] += f"\nValue: {value_match.group(1)}"

        # Extract BinOp type
        if comp_type == "BinOp" or comp_type == "UnaryOp":
            op_match = re.search(r"op: (\w+)\.(\w+)", comp_content)
            if op_match:
                component["label"] += f"\nOp: {op_match.group(2)}"

        # Parse ports
        ports = re.findall(
            r"Port\[(\d+)\] (\w+) \((.*?)\)(?: => | <= )?\[?(\d+)?\]? (\w+)?",
            comp_content,
        )
        for port in ports:
            port_num, port_name, port_type, target_port, target_name = port
            if port_num in port_to_component:
                continue
            # print(port_num, port_name)
            component["ports"][port_num] = {
                "name": port_name,
                "type": port_type,
                "direction": "in" if port_name.startswith("i_") else "out",
            }
            port_to_component[port_num] = comp_id

            if target_port and port_name.startswith("o_"):
                component["connections"].append((port_num, target_port))

        unconnected_ports = re.findall(r"Port\[(\d+)\] (\w+) \((.*?)\)", comp_content)
        for port in unconnected_ports:
            port_num, port_name, port_type = port
            if port_num in port_to_component:
                continue
            print(port_num, port_name)
            component["ports"][port_num] = {
                "name": port_name,
                "type": port_type,
                "direction": "in" if port_name.startswith("i_") else "out",
            }
            port_to_component[port_num] = comp_id
        components.append(component)

    # 解析 inputs 和 outputs
    inputs = []
    outputs = []

    inputs_match = re.search(r"inputs:\s*\[(.*)\]", text)
    if inputs_match:
        inputs_text = inputs_match.group(1)
        inputs = re.findall(r"Port\[(\d+)\] (\w+) \((.*?)\)", inputs_text)

    outputs_match = re.search(r"outputs:\s*\[(.*)\]", text)
    if outputs_match:
        outputs_text = outputs_match.group(1)
        outputs = re.findall(r"Port\[(\d+)\] (\w+) \((.*?)\)", outputs_text)

    return components, port_to_component, inputs, outputs


def visualize_components(text):
    components, port_to_component, inputs, outputs = parse_components(text)
    # print(inputs, outputs)
    dot = Digraph(comment="Component Visualization")

    # 添加输入节点
    if inputs:
        dot.node(
            "input_node",
            "Input",
            shape="ellipse",
            style="filled",
            fillcolor="lightblue",
        )

    # 添加输出节点
    if outputs:
        dot.node(
            "output_node",
            "Output",
            shape="ellipse",
            style="filled",
            fillcolor="lightgreen",
        )

    # Add nodes
    for comp in components:
        shape = "ellipse"
        if "Constant" in comp["type"]:
            shape = "box3d"
        elif "BinOp" in comp["type"]:
            shape = "diamond"
        elif "Scatter" in comp["type"]:
            shape = "trapezium"
        dot.node(comp["id"], comp["label"], shape=shape)

    # Add edges
    connections = set()
    for comp in components:
        for src_port, tgt_port in comp["connections"]:
            if src_port in port_to_component and tgt_port in port_to_component:
                src_comp = port_to_component[src_port]
                tgt_comp = port_to_component[tgt_port]
                if (src_comp, tgt_comp) not in connections:
                    label = f'{comp["ports"][src_port]["name"]}: {comp["ports"][src_port]["type"]}'
                    dot.edge(src_comp, tgt_comp, label)
                    connections.add((src_comp, tgt_comp))

    for port_num, port_name, port_type in inputs:
        # print(port_num, port_name, port_type)
        if port_num in port_to_component:
            comp_id = port_to_component[port_num]
            dot.edge(
                "input_node",
                comp_id,
                label=f"{port_type}",
                color="blue",
                penwidth="2.0",
            )

    for port_num, port_name, port_type in outputs:
        # print(port_num, port_name, port_type)
        for comp in components:
            for p_num, p_info in comp["ports"].items():
                if p_num == port_num and p_info["direction"] == "out":
                    dot.edge(
                        comp["id"],
                        "output_node",
                        label=f"{port_type}",
                        color="green",
                        penwidth="2.0",
                    )

    return dot


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--component_def", type=str)
    parser.add_argument("--file_path", type=str)
    args = parser.parse_args()
    if args.component_def:
        dot = visualize_components(args.component_def)
        dot.render("component_graph", view=False, format="png")
    elif args.file_path:
        with open(args.file_path, "r") as f:
            text = f.read()
        dot = visualize_components(text)
        dot.render("component_graph", view=False, format="png")
    else:
        print("Please provide either a component definition or a file path.")

```

`CCFSys2025_GraphyFlow/tests/bellman_ford.py`:

```py
# tests/dist.py
from pathlib import Path
from graphyflow.global_graph import GlobalGraph
import graphyflow.dataflow_ir as dfir
from graphyflow.lambda_func import lambda_min
from graphyflow.passes import delete_placeholder_components_pass

# 导入我们最终的生成器 API
from graphyflow.project_generator import generate_project

# ==================== 配置 =======================
KERNEL_NAME = "graphyflow"
EXECUTABLE_NAME = "graphyflow_host"
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
OUTPUT_DIR = PROJECT_ROOT / "generated_project"

# ==================== 1. 定义图算法 =======================
print("--- Defining Graph Algorithm using GraphyFlow ---")
g = GlobalGraph(
    properties={
        "node": {"distance": dfir.FloatType()},
        "edge": {"weight": dfir.FloatType()},
    }
)
edges = g.add_graph_input("edge")
pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
pdu = pdu.filter(filter_func=lambda x, y, z: z >= 0.0)
min_dist = pdu.reduce_by(
    reduce_key=lambda src_dist, dst, edge_w: dst.id,
    reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
    reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
)
updated_nodes = min_dist.map_(map_func=lambda dist, node: (lambda_min(dist, node.distance), node))

# ==================== 2. 前端处理 =======================
print("\n--- Frontend Processing ---")
dfirs = g.to_dfir()
comp_col = delete_placeholder_components_pass(dfirs[0])
print("DFG-IR generated and optimized.")

# ==================== 3. 一键生成项目！ =======================
print("\n--- Generating Full Vitis Project ---")
generate_project(
    comp_col=comp_col,
    global_graph=g,
    kernel_name=KERNEL_NAME,
    output_dir=OUTPUT_DIR,
    executable_name=EXECUTABLE_NAME,
)

print("\n========================================================")
print("     Final Showcase Completed Successfully!     ")
print(f"   Project files are located in: {OUTPUT_DIR}   ")
print("========================================================")

```