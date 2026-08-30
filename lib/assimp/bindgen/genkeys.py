import re, sys
header, out = sys.argv[1], sys.argv[2]
keys = []
for line in open(header, encoding='utf-8', errors='replace'):
    m = re.match(r'#define AI_MATKEY_([A-Z0-9_]+)\s+("(?:[^"]+)"),\s*0,\s*0\s*$', line.strip())
    if m:
        keys.append((m.group(1), m.group(2)))
lines = ['package assimp', '']
for name, value in keys:
    lines.append('MATKEY_%s :: %s' % (name.ljust(28), value))
lines += ['',
          'GetMaterialFloat :: proc "c" (mat: ^Material, key: cstring, type: u32, index: u32, out: ^_real) -> Return {',
          '\treturn GetMaterialFloatArray(mat, key, type, index, out, nil)',
          '}',
          '',
          'GetMaterialInteger :: proc "c" (mat: ^Material, key: cstring, type: u32, index: u32, out: ^i32) -> Return {',
          '\treturn GetMaterialIntegerArray(mat, key, type, index, out, nil)',
          '}',
          '']
open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
print('wrote', out, '-', len(keys), 'keys')
