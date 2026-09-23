import os
import sys
import argparse

from pathlib import Path

from yaml import safe_load

import geopandas as gpd

from rich import print

from .retrieval import AEFManager,get_aef_gee
from .utils import sample_from_landfire_geometry,sample_landfire_ims

def _add_override_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        'overrides',
        nargs='*',
        metavar='key=value',
        help='Config overrides '
    )

def _cmd_train(args: argparse.Namespace) -> None:
    pass

def _cmd_inference(args: argparse.Namespace) -> None:
    pass

def _cmd_sample(args: argparse.Namespace)->None:
    if args.dataset == 'landfire':
        sample_from_landfire_geometry(
            args.expname,
            args.geometry,
            args.group_names,
            args.samples_per_group,
            args.img_size,
            args.outfile,
            args.seed
        )
    else:
        raise NotImplementedError('Unrecognized dataset sampling type')

def _cmd_retrieve(args: argparse.Namespace)->None:
    geodf = gpd.read_file(args.geometry)
    if args.groups is None:
        args.groups = list(geodf['group'].unique())

    for group in args.groups:
        group_geo = geodf[geodf['group'] == group]
        work_dir = Path(args.work_dir) / group.replace(' ','_')
        work_dir.mkdir(parents=True,exist_ok=True)

        if args.dataset == 'aef':
            if args.source == 'source.coop':
                aef_manager = AEFManager(
                    work_dir,
                    Path(args.cache_dir),
                    args.cache_size,
                    dst_crs=args.dst_crs,
                    dst_scale=args.dst_scale,
                    year = args.year
                )
            
                aef_manager.get_aef_samples(group_geo,n_jobs=args.n_jobs,cleanup=args.clean)
            elif args.source == 'ee':
                get_aef_gee(
                    work_dir,
                    dst_crs=args.dst_crs,
                    dst_scale=args.dst_scale,
                    project=args.project,
                    year=args.year
                )
        elif args.dataset == 'fbfm40':
            sample_landfire_ims(
                work_dir,
                Path(args.cache_dir),
                geometry=group_geo,
                year=args.year,
                dataset=args.dataset,
                n_jobs=args.n_jobs
            )
    
def _build_parser(mode: str):

    parser = argparse.ArgumentParser(
        prog='fuelex'
    )
    sub = parser.add_subparsers(dest='command',required=True)

    train = sub.add_parser('train')

    _add_override_arg(train)
    train.set_defaults(func=_cmd_train)

    inference = sub.add_parser('inference')

    _add_override_arg(inference)
    inference.set_defaults(func=_cmd_inference)

    sample = sub.add_parser('sample')

    sample.add_argument('--geometry',type=str)
    sample.add_argument('--group_names',type=str)
    sample.add_argument('--img_size',type=int,default=256)
    sample.add_argument('--seed',type=int,default=2000)
    sample.add_argument('--outfile',type=str)
    sample.add_argument('--samples_per_group',type=int,default=1000)
    sample.add_argument('--dataset',type=str,default='landfire')
    sample.add_argument('--expname',type=str,default='fuelex')
    sample.add_argument('--splits',type=str,nargs='+')

    _add_override_arg(sample)
    sample.set_defaults(func=_cmd_sample)

    retrieve = sub.add_parser('retrieve')

    retrieve.add_argument('--dataset',type=str,default='FBFM40')
    retrieve.add_argument('--geometry',type=str)
    retrieve.add_argument('--groups',type=str,nargs='+')
    retrieve.add_argument('--cache-dir',type=str)
    retrieve.add_argument('--work-dir',type=str)
    retrieve.add_argument('--dst-scale',type=int,default=30)
    retrieve.add_argument('--dst-crs',type=str,default='EPSG:5070')
    retrieve.add_argument('--year',type=int,default=2025)
    retrieve.add_argument('--cache-size',type=int,default=10,help='Number of concurrent tiles to store and run image cutting jobs on.')
    retrieve.add_argument('--source',type=str,default='gee')
    retrieve.add_argument('--ee-project',type=str,default=None)
    retrieve.add_argument('--n-jobs',type=int,default=1)
    retrieve.add_argument('--clean',action='store_true')
    
    _add_override_arg(retrieve)
    retrieve.set_defaults(func=_cmd_retrieve)

    return parser

def main(argv: list[str]| None=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    overrides=[]
    if argv and argv[0] in ('train', 'inference', 'sample','retrieve'):
        rest = [argv[0]]
        for token in argv[1:]:
            if '=' in token and not token.startswith('-'):
                overrides.append(token)
            else:
                rest.append(token)
        argv = rest

    parser = _build_parser(argv[0])

    args = parser.parse_args(argv)

    if hasattr(args,'overrides'):
        args.overrides = [*args.overrides,*overrides]
    args.func(args)

if __name__ == 'main':
    main()