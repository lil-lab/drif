HTML_TEMPLATE = """
<html>
  <head>
    <style>
    body {background-color: #eeeeee;}
    .clearfix::after { content: ""; clear: both; display: table; }
    .ovf {overflow: auto; }
    .container {background-color: #ffffff; margin: 5px; padding: 5px;  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.2), 0 3px 10px 0 rgba(0, 0, 0, 0.19); }
    .novel-object-dataset-container {width: 50%; display:block; float:right;}
    .instruction-container {height: 100px; display: block; float:top;}
    .object-ref-masks-container {width: 45%; display: block;}
    .similarity-matrix-container {width: 45%; display: block;}
    .similarity-matrix {width: 600px; height: 400px;  margin: 0 auto; display:block; }
    .instruction {font-family: "Lucida Console", Monaco, monospace; position:absolute; white-space: pre; margin: 5px;}
    .chunk-overlay {color: red;}
    .ref-overlay {color: blue;}
    .caption {text-align: center;}
    .fpv-block {width: 128px; margin: 1px; float: left;}
    .fpv-block-r {width: 128px; margin: 1px; float: right;}
    .fpv-block img {width: 128px; }
    .query-block {width: 64px; margin: 1px; float: left;}
    .query {width: 64px; height: 64px;}
    .nod-names {float: left;}
    .fleft {float: left;}
    .top-down-container {float: left; width: 45%;}
    .overhead-view {width: 300px; height: 300px; margin: 0 auto; display: block;}
    .trajectories-container {display: block; width: 100%; float:left;}
    .visual-trajectory {display: block; width: 100%; float: left;}
    .traj-label {float: left; width: 15%; display: block; margin: auto 0;}
    .nod-row {}
    .rpn-similarity-matrix {width: 400px; height: 300px;  margin: 0 auto; display:block; }
    .rpn-trajectory-container {width: 100%; float:left; padding: 2px; display: flex; flex-wrap: nowrap; overflow-x: auto;}
    .rpn-trajectory-step {flex: 0 0 auto; width: 450px; margin: 2px;}
    .rpn-fpv-block {width: 128px; margin: 1px; float: left; margin: 2px; }
    .rpn-region-block {width: 50px; margin: 1px; float: left; margin: 2px; }
    .rpn-regions {width: 400px; float: left; display: block; margin: 2px; }
    .rpn-similarity-matrix-container {width: 400px; display: block; margin: 2px; }
    tr:nth-child(even) {background-color: #f2f2f2;}
    th, td {border-bottom: 1px solid #ddd;}
    th {background-color: #555555; color: white;}
    </style>
  </head>
  <body>
    <div class="instruction-container container">
      <h3>Instruction, <span class="chunk-overlay">Spurious Chunks</span>, <span class="ref-overlay">Object References</span>.</h3>
      ${INSTR}$
    </div>
  
    <div class="novel-object-dataset-container container">
      <h3>Novel Object Dataset & Recognition Masks</h3>
      ${NOD}$
    </div>

    <div class="similarity-matrix-container container">
      <h3>Text Similarity Matrix</h3>
      ${TEXT_SIMILARITY_MATRIX}$
    </div>
    
    <div class="top-down-container container">
      <h3>Overhead View / Executed Trajectory</h3>
      ${OVERHEAD_VIEW}$
    </div>
    
    <div class="dict-container container clearfix">
      <h3>Total Intrinsic Rewards</h3>
      ${INTRINSIC_REWARDS}$
    </div>
    
    <div class="rpn-trajectory-container container clearfix">
      <!--<h3>RPN Trajectory</h3>-->
       ${RPN_TRAJECTORY_STEPS}$
    </div>
  </body>
</html>

"""

HTML_RPN_STEP_TEMPLATE = """
      <div class="rpn-trajectory-step">
        <b>Timestep: ${TIMESTEP_ID}$</b>
        <div class="rpn-regions container">
          <h4>Recognized Regions</h4>
          <div class="rpn-fpv-block">
            <img src="${FPV_IMG_SRC}$"/>
            <p class="caption">FPV</p>
          </div>
          ${REGION_CROPS}$
        </div>
        <div class="rpn-similarity-matrix-container container">
          <h4>Visual Similarity Matrix</h3>
          <img class="rpn-similarity-matrix" src="${VISUAL_SIM_MATRIX_SRC}$"/>
        </div>
        <div class="rpn-similarity-matrix-container container">
          <h4>Grounding Matrix</h3>
          <img class="rpn-similarity-matrix" src="${GROUNDING_MATRIX_SRC}$"/>
        </div>
        <div class="rpn-regions container">
          <h4 class="traj-label">Object FPV Masks</h3>
          ${REGION_FPV_MASKS}$
        </div>
        <div class="rpn-regions container">
          <div class="fpv-block">
            <img src="${TOP_DOWN_SRC}$"/>
            <p class="caption">Top-Down View</p>
          </div>
          <div class="fpv-block">
            <img src="${OBJECT_MAP_SRC}$"/>
            <p class="caption">Object Map</p>
          </div>
          <div class="fpv-block">
            <img src="${VDIST_SRC}$"/>
            <p class="caption">Visitation Dist</p>
          </div>
          <div class="fpv-block">
            <img src="${ALL_OBJ_FPV_SRC}$"/>
            <p class="caption">All objects FPV</p>
          </div>
          <div class="fpv-block">
            <img src="${ALL_OBJ_W_SRC}$"/>
            <p class="caption">All objects W</p>
          </div>
        </div>
        <div class="intrinsic-rewards container clearfix">
          <h4 class="traj-label">Intrinsic Rewards</h3>
          ${INTRINSIC_REWARDS}$
        </div>
      </div>
"""

HTML_RPN_REGION_CROP_TEMPLATE = """
      <div class="rpn-region-block">
        <img src="${IMG_SRC}$"/>
        <p class="caption">${REGION_LABEL}$</p>
      </div>
"""

HTML_RPN_REGION_FPV_MASK_TEMPLATE = """
      <div class="fpv-block">
        <img src="${IMG_SRC}$"/>
        <p class="caption">${LABEL}$</p>
      </div>
"""


HTML_NOD_ROW_TEMPLATE = """
    <div class="nod-row clearfix">
        ${CONTENTS}$
    </div>
"""

HTML_NOD_ROW_LABEL_TEMPLATE = """
<p class="caption fleft">${LABEL}$</p>
"""

HTML_NOD_ROW_QUERY_BLOCK_TEMPLATE = """
<div class="query-block"><img class="query" src="${SRC_IMG}$"/></div>
"""

HTML_NOD_ROW_NAMES_TEMPLATE = """
<div class="nod-names">
  <ul>
    <li>${1}$</li>
    <li>${2}$</li>
    <li>${3}$</li>
    <li>${4}$</li>
    <li>${5}$</li>
  </ul>
</div>
"""

HTML_NOD_ROW_MASK_TEMPLATE = """
<div class="fpv-block-r">
  <img src="${IMG_SRC}$"/>
  <p class="caption">${IMG_CAPTION}$ </p>
</div>
"""

HTML_INSTR_TEMPLATE = """
<p class="instruction">${INSTR}$</p>
<p class="instruction chunk-overlay">${CHUNK_OVERLAY}$</p>
<p class="instruction ref-overlay">${REF_OVERLAY}$</p>
"""

HTML_SIMILARITY_MATRIX_TEMPLATE = """
<img class="similarity-matrix" src="${IMG_SRC}$"/>
"""

HTML_OVERHEAD_VIEW_TEMPLATE = """
<img class="overhead-view" src="${IMG_SRC}$"/>
"""

HTML_VISUAL_TRAJECTORY_TEMPLATE = """
<div class="visual-trajectory">
    <h3 class="traj-label">${LABEL}$</h3>
    ${BLOCKS}$
</div>
"""

HTML_VISUAL_TRAJECTORY_BLOCK_TEMPLATE = """
<div class="fpv-block">
  <img src="${IMG_SRC}$"/>
  <p class="caption">${CAPTION}$</p>
</div>
"""


def add_table(html, token, datadict):
    inner_html = "<table>\n"
    for k, v in datadict.items():
        row = f"    <tr><td>{k}</td><td>{v}</td></tr>\n"
        inner_html += row
    inner_html += "</table>"
    html = fill(html, token, inner_html)
    return html


def fill(html, token, inner_html):
    full_tok = "${" + str(token) + "}$"
    idx = html.find(full_tok)
    html_out = html[:idx] + inner_html + html[idx+len(full_tok):]
    return html_out


def multi_fill(html: object, tokens: object, inner_htmls: object) -> object:
    for token, inner_html in zip(tokens, inner_htmls):
        html = fill(html, token, inner_html)
    return html
