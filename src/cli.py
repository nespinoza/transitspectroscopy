"""Command-line interface: ``transitspectroscopy run config.toml``."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='transitspectroscopy',
        description='Configuration-driven transit spectroscopy reductions.',
    )
    commands = parser.add_subparsers(dest='command', required=True)
    run_parser = commands.add_parser('run', help='run the reduction described by a TOML file')
    run_parser.add_argument('config', help='run configuration (TOML)')
    run_parser.add_argument('--force', action='store_true', help='recompute every step')
    run_parser.add_argument(
        '--from-step',
        choices=['data', 'calibrate', 'extract', 'lightcurves', 'fit'],
        help='recompute this step and the following ones',
    )
    run_parser.add_argument(
        '--no-input-hashes',
        action='store_true',
        help='do not record SHA-256 of the input files (faster)',
    )
    commands.add_parser('modes', help='list the registered observing modes')
    args = parser.parse_args(argv)
    if args.command == 'modes':
        from . import instruments

        for name in instruments.list_modes():
            mode = instruments.get_mode(name)
            handler = getattr(mode.stage2, '__name__', mode.stage2)
            print(f'{name:20s} {mode.observatory:6s} {mode.kind:22s} stage2={handler}')
        return 0
    from .runner import run

    result = run(
        args.config,
        force=args.force,
        from_step=args.from_step,
        hash_inputs=not args.no_input_hashes,
    )
    print(f'Done. Outputs in {result.output}; reused steps: {", ".join(result.reused) or "none"}')
    for variant, products in result.products.items():
        print(f'  {variant}: ' + ', '.join(sorted(products)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
