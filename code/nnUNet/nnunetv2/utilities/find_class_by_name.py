import importlib
import pkgutil
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import *
from pathlib import Path # 修复path


def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, class_name):
                tr = getattr(m, class_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break

    # 如果在原始路径中找不到，尝试在 variants 目录中查找
    if "nnUNetTrainerVNeter" in current_module and not folder.endswith("variants"):
        variants_folder = join(Path(folder).parent, "variants")
        variants_module = current_module.rsplit('.', 1)[0] + ".variants"
        if isdir(variants_folder):
            return recursive_find_python_class(variants_folder, class_name, variants_module)

    return tr
