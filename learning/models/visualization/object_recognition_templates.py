OBJECT_PROPOSAL_TEMPLATE_HTML = """
<html>
  <head>
    <style>
    body {background-color: #dddddd;}
    .clearfix::after { content: ""; clear: both; display: table; }
    .container {background-color: #ffffff; margin: 5px; padding: 5px;  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.2), 0 3px 10px 0 rgba(0, 0, 0, 0.19);}
    .correct-predictions-container {display: block; float:left; }
    .wrong-predictions-container {display: block; float:left; }
    .per-obj-mistakes-container {}
    .image-block {margin: 1px; float: left;}
    td { padding: 10px;}
    th, td {border: 1px solid #dddddd; }
    .image { width: 264px; }
    </style>
  </head>
  <body>
    <div class="object-proposal-container container clearfix">
      <h3>Object Proposals</h3>
      ${OBJECT_PROPOSALS}$
    </div>
  </body>
</html>
"""

OBJECT_PROPOSAL_ROW_HTML_TEMPLATE = """
<div class="image-block"><img class="image" src="${IMG_SRC}$"/></div>
"""


def fill(html, token, inner_html):
    full_tok = "${" + str(token) + "}$"
    idx = html.find(full_tok)
    html_out = html[:idx] + str(inner_html) + html[idx+len(full_tok):]
    return html_out


def multi_fill(html: object, tokens: object, inner_htmls: object) -> object:
    for token, inner_html in zip(tokens, inner_htmls):
        html = fill(html, token, inner_html)
    return html
