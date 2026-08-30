import re, sys
d = sys.argv[1]
def fix(name, f):
    p = d + '/' + name + '.odin'
    s = open(p, encoding='utf-8').read()
    s = f(s)
    open(p, 'w', encoding='utf-8', newline='\n').write(s)
fix('defs', lambda s: s.replace('SIZE_MAX :: (~((c.size_t)0))', 'SIZE_MAX :: max(uint)')
                       .replace('ai_epsilon        :: ((_real)1e-6)', 'ai_epsilon        :: _real(1e-6)'))
stub = re.compile(r'(?m)^(Scene|FileIO|Texture|Node|String)\s*:: struct \{\}\n')
for f in ['cexport', 'cimport', 'mesh', 'metadata']:
    fix(f, lambda s: stub.sub('', s))

import glob
for p2 in glob.glob(d + '/*.odin'):
    s2 = open(p2, encoding='utf-8').read()
    a2 = 'foreign import lib "bin/assimp-vc143-mt.lib"'
    if a2 in s2 and 'when ODIN_OS' not in s2:
        b2 = ('when ODIN_OS == .Windows {' + chr(10)
              + chr(9) + 'foreign import lib "bin/assimp-vc143-mt.lib"' + chr(10)
              + '} else {' + chr(10)
              + chr(9) + 'foreign import lib "bin/libassimp.so"' + chr(10)
              + '}')
        open(p2, 'w', encoding='utf-8', newline=chr(10)).write(s2.replace(a2, b2, 1))

print('postfix done')
