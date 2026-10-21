# eval-kit (JavaScript)

Did the eval get better, or is that noise? The zero-dependency JS binding and CLI: Node 22+,
Node built-ins only.

## Install

Not on npm yet. Run it in place from a clone:

```bash
git clone https://github.com/TracineHQ/eval-kit && cd eval-kit
node js/cli.mjs --help
```

For the library alone, `js/lib/stats.mjs` is self-contained: copy it into your project.

## Command line

```bash
node js/cli.mjs compare --baseline base.json --variant new.json --target 3
node js/cli.mjs power --baseline pilot.json --target 3
```

## Library

```javascript
import { verdict, welchTTest, pairedTTest, requiredN } from './js/lib/stats.mjs';

const v = verdict([80.1, 82.4, 79.0, 83.5], [83.9, 81.2, 86.0, 84.7], { target: 3 });
// v.verdict 'cant_tell', v.reason 'underpowered', v.needed 16 runs per arm
```

See the top-level [README](https://github.com/TracineHQ/eval-kit#readme) for input formats,
exit codes, and methodology.

## License

Apache 2.0. See [LICENSE](https://github.com/TracineHQ/eval-kit/blob/main/LICENSE) and
[NOTICE](https://github.com/TracineHQ/eval-kit/blob/main/NOTICE).
