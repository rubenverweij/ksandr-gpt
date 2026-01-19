# Dummy references
import re

REFS = {
    "db10": "[link to='/aad/9027/dossier']DB10[/link]",
    "dr12": "[link to='/aad/655/dossier']DR12[/link]",
    "bcc": "[link to='/aad/9027/dossier']BCC[/link]",
    "magnefix": "[link to='/aad/318/dossier']Magenfix[/link]",
    "merlin": "[link to='/aad/1555/dossier']Merlin[/link]",
    "safe": "[link to='/aad/8825/dossier']Safe[/link]",
    "conel": "[link to='/aad/1556/dossier']Conel[/link]",
    "svs": "[link to='/aad/2061/dossier']SVS[/link]",
    "coq": "[link to='/aad/1557/dossier']Coq[/link]",
    "8djh": "[link to='/aad/10545/dossier']8DJH[/link]",
    "vd4": "[link to='/aad/10536/dossier']VD4[/link]",
    "ela12": "[link to='/aad/10535/dossier']ELA12[/link]",
    "l-sep": "[link to='/aad/10540/dossier']L-SEP[/link]",
    "rm6": "[link to='/aad/1555/dossier']RM6[/link]",
    "nxplusc": "[link to='/aad/10542/dossier']Nxplusc[/link]",
    "laagspanning": "[link to='/aad/10551/dossier']laagspanning[/link]",
    "kabelmoffen": "[link to='/aad/8827/dossier']laagspanning[/link]",
    "mms": "[link to='/aad/9026/dossier']MMS[/link]",
    "fmx": "[link to='/aad/10546/dossier']FMX[/link]",
    "xiria": "[link to='/aad/2059/dossier']Xiria[/link]",
    "capitole": "[link to='/aad/1558/dossier']Capitole[/link]",
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
