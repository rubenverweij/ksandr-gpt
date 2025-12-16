# Dummy references
import re

REFS = {
    "db10": "[link to='/aads/9027']DB10[/link]",
    "dr12": "[link to='/aads/655']DR12[/link]",
    "bcc": "[link to='/aads/9027']BCC[/link]",
    "magnefix": "[link to='/aads/318']Magenfix[/link]",
    "merlin": "[link to='/aads/1555']Merlin[/link]",
    "safe": "[link to='/aads/8825']Safe[/link]",
    "conel": "[link to='/aads/1556']Conel[/link]",
    "svs": "[link to='/aads/2061']SVS[/link]",
    "coq": "[link to='/aads/1557']Coq[/link]",
    "8djh": "[link to='/aads/10545']8DJH[/link]",
    "vd4": "[link to='/aads/10536']VD4[/link]",
    "ela12": "[link to='/aads/10535']ELA12[/link]",
    "l-sep": "[link to='/aads/10540']L-SEP[/link]",
    "rm6": "[link to='/aads/1555']RM6[/link]",
    "nxplusc": "[link to='/aads/10542']Nxplusc[/link]",
    "laagspanning": "[link to='/aads/10551']laagspanning[/link]",
    "kabelmoffen": "[link to='/aads/8827']laagspanning[/link]",
    "mms": "[link to='/aads/9026']MMS[/link]",
    "fmx": "[link to='/aads/10546']FMX[/link]",
    "xiria": "[link to='/aads/2059']Xiria[/link]",
    "capitole": "[link to='/aads/1558']Capitole[/link]",
}


def replace_patterns(text: str) -> str:
    for pattern, replacement in REFS.items():
        text = re.sub(
            re.escape(pattern),  # literal match
            replacement,
            text,
            flags=re.IGNORECASE,  # hoofdletterongevoelig
        )
    return text
