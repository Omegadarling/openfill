# SPDX-License-Identifier: Apache-2.0
"""Scene-based recommendations shared by both PhotoCraft panels."""

from openfill.remote_models import DEV_ID, LAMA_ID, QWEN21_LIGHTNING_ID, ROREM_ID

from .service import TEXTURE_REMOVAL

SCENE_PLACEHOLDER = "Choose a situation…"
SELECTION_HINT = "Include the object and any shadow you want removed. A small edge margin clears outlines."


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
