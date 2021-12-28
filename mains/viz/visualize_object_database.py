import os
import json
import imgkit
import base64


#DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/hosted_dashboards/object_databases/object_database_test_sim_seen"
#DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/hosted_dashboards/object_databases/object_database_test_sim_03"
#DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/hosted_dashboards/object_databases/object_database_test_real_seen"
DIR = "/media/clic/shelf_space/cage_workspaces/unreal_config_nl_cage_rss2020/hosted_dashboards/object_databases/object_database_test_real_unseen"

def img_to_b64(path):
    with open(path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


HTML_TEMPLATE = """
<html>
  <head/>
    <link rel="stylesheet" type="text/css"
      href="https://cdn.rawgit.com/dreampulse/computer-modern-web-font/master/fonts.css">
    <style>
      img {width: 128; height: 128; margin: 3px; }
      .obj {clear: both; padding-top: 6pt; }
      .objstr {float: left;}
      .obj img {float: left;}
      p {margin-left: 8pt; margin-bottom: 2px; margin-top: 0px; font-size: 13pt;  font-family: "Computer Modern Sans", sans-serif;}
    </style>
  <body>
    <div class="database">
      ${OBJECTS}$
    </div>
  </body>
</html>
"""

OBJECT_TEMPLATE = """
<div class="obj">
    ${IMAGES}$
    <div class="objstr">
      ${REFERENCES}$
    </div>
</div>
<br/>
"""

IMAGE_TEMPLATE = """<img src="data:;base64,${FILENAME}$" />"""
REFERENCE_TEMPLATE = """<p>${REFERENCE}$</p>"""


def fill(html, token, inner_html):
    full_tok = "${" + str(token) + "}$"
    idx = html.find(full_tok)
    html_out = html[:idx] + inner_html + html[idx+len(full_tok):]
    return html_out


def viz_object_database():
    refs_file = os.path.join(DIR, "object_references.json")
    imgs_dir = os.path.join(DIR, "c_objects")
    out_file = os.path.join(DIR, "index.html")

    with open(refs_file, "r") as fp:
        obj_refs = json.load(fp)

    objs_html = ""

    for key, ref_list in obj_refs.items():
        references_html = "\n".join([fill(REFERENCE_TEMPLATE, "REFERENCE", ref) for ref in ref_list])
        image_fnames = os.listdir(os.path.join(imgs_dir, key))
        image_relpaths = [os.path.join("c_objects", key, imgn) for imgn in image_fnames]
        image_data = [img_to_b64(os.path.join(DIR, relpath)) for relpath in image_relpaths]

        images_html = "\n".join([fill(IMAGE_TEMPLATE, "FILENAME", rp) for rp in image_data])

        obj_html = fill(fill(OBJECT_TEMPLATE, "IMAGES", images_html), "REFERENCES", references_html)
        objs_html += obj_html

    out_html = fill(HTML_TEMPLATE, "OBJECTS", objs_html)

    with open(out_file, "w") as fp:
        fp.write(out_html)

    os.chdir(DIR)
    imgkit.from_string(out_html, 'overview.jpg', options={'width': '920'})


if __name__ == "__main__":
    viz_object_database()