# SPDX-License-Identifier: Apache-2.0
"""Scene-based recommendations shared by both PhotoCraft panels."""

from openfill.remote_models import DEV_ID, LAMA_ID, QWEN21_LIGHTNING_ID, ROREM_ID

TEXTURE_REMOVAL = "photocraft-texture"

AUTO = "Automatic from selection"
SCENE_PLACEHOLDER = "Manual method"
SELECTION_HINT = "Include the object and any shadow you want removed. A small edge margin clears outlines."


def recommend(features, available, existing_server):
    """Rank tested repair methods using visible cues. Never infer the identity of hidden content."""
    local = (TEXTURE_REMOVAL,)
    lama = (LAMA_ID, "lama-big", "mat-places512")
    rorem = (ROREM_ID, "rorem-sdxl")
    qwen = (QWEN21_LIGHTNING_ID,)
    fibo = (DEV_ID,)
    f = features
    local_tip = "Local is the available fallback. Check for repeated texture or broken edges in this selection."
    if f["samples"] < 7:
        ranking = (*lama, *rorem, *local)
        tip = "Little opaque background is visible around the selection. Leave some context and inspect the repair."
    elif f["coverage"] < 0.10 and f["detail"] < 0.04 and f["range"] < 0.15 and f["jump"] < 0.09 and f["bend"] > 0.17:
        ranking = local
        tip = "A quiet rim surrounds this small selection, with stronger features farther out. Local can fill from the rim."
    elif f["detail"] <= 0.02 and f["edge"] <= 0.035 and f["jump"] < 0.16:
        if f["bend"] < 0.025:
            ranking = local
            tip = "The surrounding pixels change gradually. Local can continue this lighting. Include any cast shadow."
        elif f["range"] > 0.20:
            ranking = (*rorem, *lama, *local)
            tip = "Soft lighting curves across a form near the selection. Inspect its outline after removal."
        else:
            ranking = (*lama, *rorem, *local)
            tip = "There is faint texture beyond the selection. Check that the fill carries it across the repair."
            local_tip = "Local can use the similar colors around the lasso to sample background texture. Check that the pattern continues across the repair."
    elif f["chroma"] < 0.10 and f["edge"] > 0.32 and f["range"] > 0.52:
        ranking = (*lama, *rorem, *local)
        tip = "There is fine, high-contrast detail here. If it is lettering, use a reference to restore the exact text."
    elif f["coverage"] >= 0.20 and f["cool"] >= 0.80 and f["aspect"] < 2:
        ranking = (*fibo, *lama, *rorem, *local)
        tip = "A broad selection crosses blue-green texture. Check how the fill continues the surrounding lighting."
    elif f["aspect"] >= 1.8 and f["range"] > 0.55 and f["highlights"] > 0:
        ranking = (*rorem, *lama, *local)
        tip = "This elongated area crosses bright and dark surfaces. Inspect their edges where they meet the repair."
    elif f["coverage"] >= 0.12 and f["aspect"] < 1.8 and f["jump"] > 0.18:
        ranking = (*qwen, *lama, *rorem, *local)
        tip = "A substantial area spans distinct edges. Check reconstructed shapes against a reference if precision matters."
    else:
        ranking = (*lama, *rorem, *fibo, *local)
        tip = "The surrounding pixels contain edges or texture. A masked repair can continue nearby detail."
    ids = set(available) | {TEXTURE_REMOVAL}
    method = next(i for i in ranking if i in ids)
    if method == TEXTURE_REMOVAL and ranking != local:
        tip = local_tip
    default = next((i for i in ("lama-big", "mat-places512") if i in ids), None)
    display = "default" if not existing_server and method == default else method
    return {"method": method, "displayMethod": display, "tip": tip}


def removal_scenes(available, existing_server):
    """Return choices that resolve to the actual Method menu on this source."""
    ids = set(available)
    default = None if existing_server else next((i for i in ("lama-big", "mat-places512") if i in ids), None)
    lama = next((i for i in (LAMA_ID, "lama-big") if i in ids), None)
    rorem = next((i for i in (ROREM_ID, "rorem-sdxl") if i in ids), None)
    qwen = QWEN21_LIGHTNING_ID if QWEN21_LIGHTNING_ID in ids else None
    fibo = DEV_ID if DEV_ID in ids else None
    scenes = []

    def add(scene, method, tip):
        scenes.append({"scene": scene, "method": "default" if default and method == default else method, "tip": tip})

    add(
        "Smooth studio backdrop",
        TEXTURE_REMOVAL,
        "Local continues smooth lighting. Include the full cast shadow in your selection.",
    )
    add(
        "Wire or repeating pattern",
        lama,
        "LaMa continued paving joints and cleared a cable across a rocket. Include the full width of the wire."
        if lama
        else "LaMa gave the cleanest wire and paving repairs. Choose a source with LaMa for this case.",
    )
    add(
        "Object crossing a shape",
        qwen or lama or rorem,
        "Qwen rebuilt a wheel and an eye. Inspect the shape against a reference when its exact geometry matters."
        if qwen
        else "LaMa can continue nearby edges. Inspect hidden geometry at full size."
        if lama
        else "Try RORem and inspect the shape. Missing wheel details distorted in testing."
        if rorem
        else "Qwen Image 2.1 Lightning gave the best wheel repair. Choose a source offering it for this case.",
    )
    add(
        "Clutter on texture",
        lama or TEXTURE_REMOVAL,
        "Try LaMa for small objects on texture. Include contact shadows."
        if lama
        else "Local samples nearby texture. Include contact shadows and check the detail at full size.",
    )
    add(
        "Reflective object",
        rorem or lama,
        "RORem removed a reflective spoon where it crossed a cup and saucer."
        if rorem
        else "LaMa removed a spoon while continuing the saucer and cup edges."
        if lama
        else "RORem and LaMa performed best on a reflective spoon. Choose a source offering either model.",
    )
    add(
        "Water and bubbles",
        fibo or lama or rorem or TEXTURE_REMOVAL,
        "FIBO extended the bubbles behind a diver. Check for invented fish or changes in lighting."
        if fibo
        else "Try LaMa to continue the surrounding water texture."
        if lama
        else "Try RORem for the underwater background. Check light and bubbles across the selection."
        if rorem
        else "Local suits small distractions in open water. Large selections across bubbles can leave a patch.",
    )
    add(
        "Covered lettering",
        None,
        "Exact hidden letters need a reference or retyping. You can still choose a method to repair the surrounding paper.",
    )
    return scenes
