#!/usr/bin/env python

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, type=str, help='path to input video')
    parser.add_argument('--image', required=True, type=str, help='path to output illustration')
    return parser.parse_args()


def main():
    # pylint: disable=unused-variable
    args = get_args()


if __name__ == '__main__':
    main()
