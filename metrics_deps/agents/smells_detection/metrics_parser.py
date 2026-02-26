import pandas as pd
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

class MetricsParser:
    
    @staticmethod
    def normalize_columns(df):
        old_cols = df.columns
        new_cols = []
        mapping = {}

        for c in old_cols:
            clean = c.strip().replace("\ufeff", "")
            normalized = clean.lower().replace(" ", "_")
            new_cols.append(normalized)
            mapping[normalized] = clean

        df.columns = new_cols
        return df, mapping

    @staticmethod
    def parse_class_metrics(row, project_path):
        return {
            "package": row["package"],
            "class": row["class"],
            "file": row.get("file")[row.get("file").index(project_path):] if row.get("file") else "",
            "metrics": {
                "nof": int(row.get("nof", 0)),
                "nopf": int(row.get("nopf", 0)),
                "nom": int(row.get("nom", 0)),
                "nopm": int(row.get("nopm", 0)),
                "loc": int(row.get("loc", 0)),
                "wmc": int(row.get("wmc", 0)),
                "nc": int(row.get("nc", 0)),
                "dit": int(row.get("dit", 0)),
                "lcom": float(row.get("lcom", 0)),
                "fanin": int(row.get("fan-in", 0)),
                "fanout": int(row.get("fan-out", 0)),
            },
            #"methods": [],
            "dependencies": []
        }

    @staticmethod
    def group_classes_by_package(class_rows):
        packages_dict = defaultdict(list)

        for cls in class_rows:
            package = cls.get("package", "default_package")
            packages_dict[package].append(cls)

        packages_list = []
        for pkg_name, classes in packages_dict.items():
            pkg_metrics = {
                "num_classes": len(classes),
                "loc": sum(c["metrics"]["loc"] for c in classes),
                "Ce": 0,
                "Ca": 0,
            }

            packages_list.append({
                "package": pkg_name,
                "metrics": pkg_metrics,
                "classes": classes,
                "dependencies": []
            })

        return packages_list

    @staticmethod
    def classname_to_package(class_name):
        parts = class_name.split(".")
        while parts and parts[-1][0].isupper():
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def parser_dependencies(graph_path):
        tree = ET.parse(graph_path)
        root = tree.getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

        package_dependencies = defaultdict(set)
        class_dependencies = defaultdict(set)

        for edge in root.findall(".//g:edge", ns):
            source_pkg = MetricsParser.classname_to_package(edge.attrib["source"])
            target_pkg = MetricsParser.classname_to_package(edge.attrib["target"])

            if source_pkg != target_pkg:
                package_dependencies[source_pkg].add(target_pkg)

            source_class = edge.attrib["source"]
            target_class = edge.attrib["target"]

            if source_class != target_class:
                class_dependencies[source_class].add(target_class)

        package_dependencies = {k: list(v) for k, v in package_dependencies.items()}
        class_dependencies = {k: list(v) for k, v in class_dependencies.items()}

        return package_dependencies, class_dependencies

    @staticmethod
    def calculate_afferent_coupling(package_dependencies):
        afferent = defaultdict(int)
        for source_pkg, targets in package_dependencies.items():
            for target_pkg in targets:
                afferent[target_pkg] += 1
        return dict(afferent)
    
    '''@staticmethod
    def attach_dependencies(packages, package_dependencies, class_dependencies):
        package_index = {pkg["package"]: pkg for pkg in packages}
        
        valid_classes = {f'{pkg["package"]}.{cls["class"]}' for pkg in packages for cls in pkg["classes"]}

        for source_pkg, targets in package_dependencies.items():
            if source_pkg in package_index:
                valid_targets = [t for t in targets if t in package_index]
                package_index[source_pkg]["dependencies"] = valid_targets
                package_index[source_pkg]["metrics"]["Ce"] = len(valid_targets)

        afferent_coupling = defaultdict(int)
        for source_pkg, targets in package_dependencies.items():
            for target_pkg in targets:
                if target_pkg in package_index and source_pkg in package_index:
                    afferent_coupling[target_pkg] += 1

        for pkg_name, pkg in package_index.items():
            pkg["metrics"]["Ca"] = afferent_coupling.get(pkg_name, 0)

        for pkg in packages:
            for cls_obj in pkg["classes"]:
                class_name = f'{pkg["package"]}.{cls_obj["class"]}'
                raw_deps = class_dependencies.get(class_name, [])
                cls_obj["dependencies"] = [dep for dep in raw_deps if dep in valid_classes]

        return packages'''

    @staticmethod
    def attach_dependencies(packages: list[dict], package_dependencies: dict, class_dependencies: dict):
        package_index = {pkg["package"]: pkg for pkg in packages}

        valid_classes = {
            f'{pkg["package"]}.{cls["class"]}'
            for pkg in packages
            for cls in pkg.get("classes", [])
            if pkg.get("package") and cls.get("class")
        }

        # -------------------------
        # 1) Build outgoing + incoming sets (package level)
        # -------------------------
        outgoing_by_package = defaultdict(set)
        incoming_by_package = defaultdict(set)

        for source_pkg, targets in package_dependencies.items():
            if source_pkg not in package_index:
                continue

            valid_targets = [t for t in targets if t in package_index and t != source_pkg]
            for t in valid_targets:
                outgoing_by_package[source_pkg].add(t)
                incoming_by_package[t].add(source_pkg)

        # -------------------------
        # 2) Compute Ca/Ce/I for ALL packages
        # -------------------------
        def calc_I(ca: int, ce: int) -> float:
            denom = ca + ce
            return (ce / denom) if denom > 0 else 0.0  # escolha comum quando não há acoplamento

        for pkg_name, pkg in package_index.items():
            ce = len(outgoing_by_package.get(pkg_name, set()))
            ca = len(incoming_by_package.get(pkg_name, set()))
            I  = calc_I(ca, ce)

            pkg.setdefault("metrics", {})
            pkg["metrics"]["Ce"] = ce
            pkg["metrics"]["Ca"] = ca
            pkg["metrics"]["I"]  = round(I, 2)

        # helper: return Ca/Ce/I for a referenced package (safe defaults)
        def pkg_ref(pkg_name: str) -> dict:
            ref = package_index.get(pkg_name)
            if not ref:
                return {"package": pkg_name, "Ca": 0, "Ce": 0, "I": 0.0}

            m = ref.get("metrics", {})
            ca = int(m.get("Ca", 0))
            ce = int(m.get("Ce", 0))
            I  = float(m.get("I", calc_I(ca, ce)))
            return {"package": pkg_name, "Ca": ca, "Ce": ce, "I": round(I, 2)}

        # -------------------------
        # 3) Attach enriched lists for each package
        # -------------------------
        for pkg_name, pkg in package_index.items():
            deps = sorted(outgoing_by_package.get(pkg_name, set()))
            dents = sorted(incoming_by_package.get(pkg_name, set()))

            # Each item contains Ca/Ce/I OF THAT ITEM PACKAGE
            pkg["dependencies"] = [pkg_ref(t) for t in deps]
            pkg["dependents"]   = [pkg_ref(s) for s in dents]

        # -------------------------
        # 4) Keep your class-level deps/dependents (unchanged)
        # -------------------------
        incoming_by_class = defaultdict(set)

        for pkg in packages:
            for cls_obj in pkg.get("classes", []):
                class_name = f'{pkg["package"]}.{cls_obj["class"]}'
                raw_deps = class_dependencies.get(class_name, [])
                outgoing = [dep for dep in raw_deps if dep in valid_classes]

                cls_obj["dependencies"] = outgoing
                cls_obj.setdefault("dependents", [])

                for dep in outgoing:
                    incoming_by_class[dep].add(class_name)

        for pkg in packages:
            for cls_obj in pkg.get("classes", []):
                class_name = f'{pkg["package"]}.{cls_obj["class"]}'
                cls_obj["dependents"] = sorted(incoming_by_class.get(class_name, set()))

        return packages

    @staticmethod
    def parse_method_metrics(row):
        def to_int(v, default=0):
            try:
                return int(v)
            except Exception:
                return default

        is_test_raw = row.get("istest", 0)
        try:
            is_test = bool(int(is_test_raw))
        except Exception:
            is_test = bool(is_test_raw)

        return {
            "method_name": row.get("method", row.get("method_name")),
            "metrics": {
                "loc": to_int(row.get("loc", 0), 0),
                "cc": to_int(row.get("cc", 0), 0),
                "pc": to_int(row.get("pc", 0), 0),
            },
        }

    @staticmethod
    def attach_methods_to_classes(packages, method_rows):
        class_index = {}
        for pkg in packages:
            for cls in pkg["classes"]:
                cls.setdefault("methods", [])
                class_index[(pkg["package"], cls["class"])] = cls

        not_attached = 0

        for row in method_rows:
            pkg_name = row.get("package")
            cls_name = row.get("class")

            if pkg_name is None or cls_name is None:
                not_attached += 1
                continue

            key = (pkg_name, cls_name)
            if key in class_index:
                class_index[key]["methods"].append(MetricsParser.parse_method_metrics(row))
            else:
                not_attached += 1

        if not_attached:
            print(f"Warning: {not_attached} métodos não foram anexados (package/class não encontrados).")

        return packages