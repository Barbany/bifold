import xml.etree.ElementTree as ET


class XMLModel:
    def __init__(self, xml_file):
        self.path = xml_file
        self.tree = ET.parse(self.path)
        self.cloth = next(self.tree.iter("flexcomp"))

    def save_changes_to_file(self, output_file=None):
        if output_file is None:
            output_file = self.path
        with open(output_file, "wb") as f:
            self.tree.write(f, encoding="utf-8")

    def modify_params(self, params):
        # Keys of params either of the form
        # "key" or "key_subkey" (underscore is important)
        # e.g., "damping" or "joint_damping"
        self.modify_params_v_3_or_more(params)

    def modify_params_v_3_or_more(self, params):
        for k, val in params.items():
            if "_" in k:
                *subelements, subkey = k.split("_")
                root = [self.cloth]
                for subelement in subelements:
                    if len(root) > 1:
                        for r in root:
                            if r.get("key") == subelement:
                                root = [r]
                                break
                    elif len(root) == 1:
                        root = root[0].findall(subelement)
                    else:
                        raise ValueError(f"Cannot modify {k}: Got root {root}")
                assert len(root) == 1, f"Found non-unique element for {k}: Got root {root}"
                root[0].set(subkey, str(val))
            else:
                self.cloth.set(k, str(val))
        self.save_changes_to_file()

    def modify_params_v_less_than_3(self, params):
        for k, val in params.items():
            if "_" in k:
                subelement, subkey = k.split("_")
                find_output = self.cloth.find(subelement)
                assert find_output is not None, f"Could not find {subelement} in {k}"
                find_output.set(subkey, str(val))
            else:
                self.cloth.set(k, str(val))
        self.save_changes_to_file()

    def change_texture(self, texture_file):
        for text in self.tree.iter("texture"):
            if text.attrib["name"] == "cloth_texture":
                text.set("file", texture_file)
                return
        raise ValueError("Could not change texture")

    def get_cloth_size(self):
        num_rows, num_cols, _ = map(int, self.cloth.attrib["count"].split())
        return num_rows, num_cols

    def get_mesh_ids(self, model):
        return model.flex_vertbodyid
