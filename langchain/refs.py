# Dummy references
import re

REFS = {
    "db10": "[link to='/aads/9027/dossier']DB10[/link]",
    "dr12": "[link to='/aads/655/dossier']DR12[/link]",
    "bcc": "[link to='/aads/9027/dossier']BCC[/link]",
    "magnefix": "[link to='/aads/318/dossier']Magenfix[/link]",
    "merlin": "[link to='/aads/1555/dossier']Merlin[/link]",
    "safe": "[link to='/aads/8825/dossier']Safe[/link]",
    "conel": "[link to='/aads/1556/dossier']Conel[/link]",
    "svs": "[link to='/aads/2061/dossier']SVS[/link]",
    "coq": "[link to='/aads/1557/dossier']Coq[/link]",
    "8djh": "[link to='/aads/10545/dossier']8DJH[/link]",
    "vd4": "[link to='/aads/10536/dossier']VD4[/link]",
    "ela12": "[link to='/aads/10535/dossier']ELA12[/link]",
    "l-sep": "[link to='/aads/10540/dossier']L-SEP[/link]",
    "rm6": "[link to='/aads/1555/dossier']RM6[/link]",
    "nxplusc": "[link to='/aads/10542/dossier']Nxplusc[/link]",
    "laagspanning": "[link to='/aads/10551/dossier']laagspanning[/link]",
    "kabelmoffen": "[link to='/aads/8827/dossier']laagspanning[/link]",
    "mms": "[link to='/aads/9026/dossier']MMS[/link]",
    "fmx": "[link to='/aads/10546/dossier']FMX[/link]",
    "xiria": "[link to='/aads/2059/dossier']Xiria[/link]",
    "capitole": "[link to='/aads/1558/dossier']Capitole[/link]",
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
