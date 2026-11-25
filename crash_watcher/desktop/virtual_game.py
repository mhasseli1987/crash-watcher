"""Standalone fictional-credit simulator. No screen, mouse or betting adapters."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json


def number(value, minimum=Decimal('1')):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('عدد معتبر وارد کنید') from None
    if not result.is_finite() or result < minimum:
        raise ValueError('عدد خارج از بازه است')
    return result


def color(value):
    return 'RED' if value < 2 else 'GREEN' if value < 10 else 'YELLOW'


@dataclass(frozen=True)
class Rule:
    action: str
    target: Decimal | None


@dataclass(frozen=True)
class Plan:
    rules: tuple[Rule, Rule, Rule]
    based_on: int
    reason: str
    horizon: int = 8

    def choose(self, previous, current_round):
        if not self.based_on <= current_round < self.based_on + self.horizon:
            return Rule('SKIP', None)
        return self.rules[('RED', 'GREEN', 'YELLOW').index(color(previous))]


RULE_SCHEMA = {'type': 'object', 'properties': {
    'action': {'type': 'string', 'enum': ['PLAY', 'SKIP']},
    'cashout': {'type': ['number', 'null'], 'minimum': 1.4, 'maximum': 1.9}},
    'required': ['action', 'cashout'], 'additionalProperties': False}
PLAN_SCHEMA = {'type': 'object', 'properties': {
    **{c: RULE_SCHEMA for c in ('RED', 'GREEN', 'YELLOW')},
    'reason': {'type': 'string', 'maxLength': 160}},
    'required': ['RED', 'GREEN', 'YELLOW', 'reason'], 'additionalProperties': False}


def plan_prompt(history):
    return (
        'This is an OFFLINE toy simulator using fictional credits only. Create a conditional '
        'policy for the next eight simulated rounds, not a prediction for one particular round. '
        'Each branch depends ONLY on the most recently CLOSED round: RED <2, GREEN >=2 and <10, '
        'YELLOW >=10. Choose PLAY or SKIP for each branch. PLAY needs cashout between 1.40 and '
        '1.90; SKIP needs null. Future test outcomes are hidden from you. Past outcomes do not '
        'prove predictability. Do not claim guaranteed recovery. Return the exact JSON schema; '
        'reason must be a short Persian sentence.\n'
        + json.dumps({'observed_oldest_to_newest': [str(v) for v in history[-40:]],
                      'colors_oldest_to_newest': [color(v) for v in history[-40:]],
                      'schema': PLAN_SCHEMA}, ensure_ascii=False)
    )


def parse_plan(raw, based_on):
    # Full JSON only; do not accept truncated or partially valid policies.
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError('پاسخ مدل JSON کامل نیست') from None
    if not isinstance(data, dict) or not isinstance(data.get('reason'), str):
        raise ValueError('توضیح مدل معتبر نیست')
    rules = []
    for name in ('RED', 'GREEN', 'YELLOW'):
        item = data.get(name)
        if not isinstance(item, dict) or item.get('action') not in ('PLAY', 'SKIP'):
            raise ValueError('هر سه حالت رنگ باید مشخص باشد')
        raw_target = item.get('cashout')
        target = None
        if item['action'] == 'PLAY':
            if type(raw_target) not in (int, float):
                raise ValueError('ضریب خروج باید عدد باشد')
            target = number(raw_target, Decimal('1.40'))
            if target > Decimal('1.90'):
                raise ValueError('ضریب خروج باید بین ۱٫۴۰ و ۱٫۹۰ باشد')
        elif raw_target is not None:
            raise ValueError('حالت رد کردن دست نباید ضریب خروج داشته باشد')
        rules.append(Rule(item['action'], target))
    return Plan(tuple(rules), based_on, data['reason'])


@dataclass(frozen=True)
class Entry:
    round_id: int
    stake: Decimal
    target: Decimal
    level: int
    plan_version: int


class VirtualGame:
    """Commands apply to this in-memory game only. Exact decimal accounting."""
    def __init__(self, history, base='10', credits='10000', max_doubles=8):
        self.history = [number(v) for v in history]
        if not self.history:
            raise ValueError('حداقل یک ضریب تاریخی لازم است')
        self.base = number(base, Decimal('0.01'))
        self.balance = number(credits, self.base)
        if type(max_doubles) is not int or not 0 <= max_doubles <= 8:
            raise ValueError('سقف دوبل باید بین صفر و هشت باشد')
        self.max_doubles = max_doubles
        self.total = self.cycle = Decimal('0')
        self.level = self.rounds = 0
        self.plan = None
        self.entry = None
        self.open_round = False
        self.exited = False
        self.last_live = Decimal('1')
        self.stopped = ''

    def install(self, plan):
        if plan.based_on > self.rounds or self.rounds >= plan.based_on + plan.horizon:
            return False
        if self.plan and plan.based_on < self.plan.based_on:
            return False
        self.plan = plan
        return True

    def begin(self):
        if self.open_round:
            raise ValueError('دور قبلی هنوز تمام نشده است')
        self.open_round, self.exited = True, False
        self.last_live = Decimal('1')
        self.entry = None
        if self.stopped or self.plan is None:
            return None
        decision = self.plan.choose(self.history[-1], self.rounds)
        if decision.action != 'PLAY':
            return None
        stake = self.base * 2 ** self.level
        if stake > self.balance:
            self.stopped = 'اعتبار مجازی کافی نیست'
            return None
        self.entry = Entry(self.rounds + 1, stake, decision.target, self.level, self.plan.based_on)
        self.balance -= stake
        return self.entry

    def tick(self, live):
        value = number(live)
        if not self.open_round or value < self.last_live:
            raise ValueError('ضریب زندهٔ دور معتبر نیست')
        self.last_live = value
        if self.entry and not self.exited and value >= self.entry.target:
            self.exited = True
            # This local simulator executes exactly at the planned threshold.
            self.balance += self.entry.stake * self.entry.target
            return True
        return False

    def finish(self, final):
        value = number(final)
        if not self.open_round or value < self.last_live:
            raise ValueError('نتیجهٔ دور معتبر نیست یا قبلاً ثبت شده')
        entry = self.entry
        delta = Decimal('0')
        recovered = False
        outcome = 'SKIP'
        if entry:
            outcome = 'CASHOUT' if self.exited else 'LOSS'
            delta = entry.stake * (entry.target - 1) if self.exited else -entry.stake
            self.total += delta
            self.cycle += delta
            if self.cycle >= 0:
                self.level = 0
                self.cycle = Decimal('0')
                recovered = True
            elif not self.exited:
                if self.level < self.max_doubles:
                    self.level += 1
                else:
                    self.stopped = 'سقف هشت دوبل رسیده؛ چرخه با زیان تسویه و متوقف شد'
            # A partial win retains the stake level; only a loss doubles it.
        self.rounds += 1
        self.history.append(value)
        self.open_round, self.entry = False, None
        return {'round': self.rounds, 'final': value, 'entry': entry, 'outcome': outcome,
                'delta': delta, 'cycle': self.cycle, 'total': self.total,
                'level': self.level, 'recovered': recovered, 'stopped': self.stopped}
