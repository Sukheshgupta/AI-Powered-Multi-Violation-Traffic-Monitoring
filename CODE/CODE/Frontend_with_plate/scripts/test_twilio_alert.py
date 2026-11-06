import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import app


class TestUser:
    def __init__(self, mobile):
        self.mobile = mobile


def main():
    parser = argparse.ArgumentParser(description='Send a test ViolaTracker Twilio alert.')
    parser.add_argument('mobile', help='Destination number in E.164 format, for example +919876543210')
    parser.add_argument('--plate', default='TEST1234', help='Plate text to include in the alert')
    parser.add_argument(
        '--channel',
        choices=['sms', 'whatsapp', 'both'],
        default=None,
        help='Override TWILIO_CHANNEL for this test only.',
    )
    parser.add_argument(
        '--violation',
        action='append',
        default=None,
        help='Violation label. Can be passed more than once.',
    )
    args = parser.parse_args()

    violations = args.violation or ['Not Wearing Helmet']
    if args.channel:
        app.TWILIO_CHANNEL = args.channel
    ok, note = app.send_violation_sms(TestUser(args.mobile), args.plate, violations)
    print(note or ('Alert sent.' if ok else 'Alert not sent.'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
