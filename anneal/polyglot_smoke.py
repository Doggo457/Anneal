"""Smoke programs for the six gate languages (scripts/setup_toolchains.py --check, tests/test_polyglot.py).

Each language gets the same small set: fast input of 200,000 64-bit numbers, line-by-line input, deep recursion (the
256 MiB stacks for C++ and Rust, -Xss for Java), a wrong answer, an endless loop, a syntax error, a crash, and on
Windows an attempt to read the loop's config and write outside the work folder, which must fail. Some programs also
check what the judge promises in its prompts: ONLINE_JUDGE, Java classes not named Main or inside a package, ES
module syntax in JavaScript, enums and interfaces in TypeScript. Each program has the verdict it must get.
"""

import os

from anneal import polyglot

N_BIG = 200_000
BIG_VALUES = [((i * 7919) % 1_000_003) * 1_000_000 for i in range(N_BIG)]
SUM_CASES = [("3\n1 2 3\n", "6\n"),
             (f"{N_BIG}\n" + " ".join(map(str, BIG_VALUES)) + "\n", f"{sum(BIG_VALUES)}\n")]
LINES_CASES = [("2\nhello big world\na b\n", "world big hello\nb a\n"),
               ("3\n  one  \nx y z w\nsame same\n", "one\nw z y x\nsame same\n")]
PAIR_CASES = [("2 3\n", "5\n"), ("-7 7\n", "0\n")]
DEPTH = {"cpp": 1_000_000, "rust": 1_000_000, "go": 1_000_000, "java": 300_000, "javascript": 5_000,
         "typescript": 5_000}


def depth_cases(lang):
    d = DEPTH[lang]
    return [("10\n", "5\n"), (f"{d}\n", f"{(d + 1) // 2}\n")]


def _paths(lang):
    """The two paths the isolation program tries: the loop config (read) and a file beside the scratch folder
    (write). Spelled for the language's string literals."""
    read = os.path.join(polyglot.ROOT, "experiments", "loop.yaml")
    write = os.path.join(polyglot.ROOT, "sandbox", f"escape_{lang}.txt")
    if lang in ("go", "rust"):
        return read, write
    return read.replace("\\", "\\\\"), write.replace("\\", "\\\\")


PROGRAMS = {
    "cpp": {
        "sum64": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    long long s = 0;
    for (int i = 0; i < n; i++) { long long a; cin >> a; s += a; }
    cout << s << "\n";
}
''',
        "lines": r'''#include <bits/stdc++.h>
using namespace std;
int main() {
    string line;
    getline(cin, line);
    int t = stoi(line);
    while (t--) {
        getline(cin, line);
        istringstream ss(line);
        vector<string> w;
        string x;
        while (ss >> x) w.push_back(x);
        reverse(w.begin(), w.end());
        for (size_t i = 0; i < w.size(); i++) cout << (i ? " " : "") << w[i];
        cout << '\n';
    }
}
''',
        "recursion": r'''#include <bits/stdc++.h>
using namespace std;
long long f(int n) {
    volatile char pad[64];
    pad[n & 63] = (char)n;
    if (n == 0) return 0;
    long long r = f(n - 1);
    return r + (pad[n & 63] & 1);
}
int main() { int n; cin >> n; cout << f(n) << "\n"; }
''',
        "online_judge": r'''#include <bits/stdc++.h>
int main() {
#ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
#endif
    long long a, b;
    std::cin >> a >> b;
    std::cout << a + b << "\n";
}
''',
        "wrong": '#include <cstdio>\nint main() { printf("42\\n"); }\n',
        "loop": "int main() { volatile unsigned long long x = 0; for (;;) x++; }\n",
        "syntax": "int main( { return 0 }\n",
        "crash": "int main() { return 3; }\n",
        "isolation": r'''#include <cstdio>
int main() {
    FILE* f = fopen("READ", "rb");
    FILE* g = fopen("WRITE", "wb");
    printf("%s\n", (f || g) ? "ESCAPED" : "DENIED");
}
''',
    },
    "java": {
        "sum64": r'''import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;
        int n = -1;
        long s = 0;
        String line;
        while ((line = br.readLine()) != null) {
            st = new StringTokenizer(line);
            while (st.hasMoreTokens()) {
                long v = Long.parseLong(st.nextToken());
                if (n < 0) n = (int) v; else s += v;
            }
        }
        System.out.println(s);
    }
}
''',
        "lines": r'''import java.io.*;
import java.util.*;

public class Solution {
    static class Rev {
        static String of(String line) {
            List<String> w = new ArrayList<>(Arrays.asList(line.trim().split("\\s+")));
            Collections.reverse(w);
            return String.join(" ", w);
        }
    }

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine().trim());
        StringBuilder sb = new StringBuilder();
        while (t-- > 0) sb.append(Rev.of(br.readLine())).append('\n');
        System.out.print(sb);
    }
}
''',
        "recursion": r'''import java.util.*;

class Helper {
    static int bit(int n) { return n & 1; }
}

class Main {
    static long f(int n) {
        if (n == 0) return 0;
        long r = f(n - 1);
        return r + Helper.bit(n);
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println(f(sc.nextInt()));
    }
}
''',
        "online_judge": r'''package contest.solution;

import java.io.*;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        InputStream in = System.getProperty("ONLINE_JUDGE") != null ? System.in : new FileInputStream("input.txt");
        Scanner sc = new Scanner(in);
        long a = sc.nextLong(), b = sc.nextLong();
        System.out.println(a + b);
    }
}
''',
        "wrong": "public class Main { public static void main(String[] a) { System.out.println(42); } }\n",
        "loop": "public class Main { public static void main(String[] a) { long x = 0; while (true) { x++; } } }\n",
        "syntax": "public class Main { public static void main(String[] a) { int x = ; } }\n",
        "crash": "public class Main { public static void main(String[] a) { int[] x = new int[1]; "
                 "System.out.println(x[2]); } }\n",
        "isolation": r'''import java.nio.file.*;

public class Main {
    public static void main(String[] args) {
        boolean escaped = false;
        try { Files.readAllBytes(Paths.get("READ")); escaped = true; } catch (Exception e) { }
        try { Files.write(Paths.get("WRITE"), new byte[]{1}); escaped = true; } catch (Exception e) { }
        System.out.println(escaped ? "ESCAPED" : "DENIED");
    }
}
''',
    },
    "go": {
        "sum64": r'''package main

import (
    "bufio"
    "fmt"
    "os"
)

func main() {
    r := bufio.NewReaderSize(os.Stdin, 1<<20)
    var n int
    fmt.Fscan(r, &n)
    var s int64
    for i := 0; i < n; i++ {
        var a int64
        fmt.Fscan(r, &a)
        s += a
    }
    fmt.Println(s)
}
''',
        "lines": r'''package main

import (
    "bufio"
    "fmt"
    "os"
    "strconv"
    "strings"
)

func main() {
    sc := bufio.NewScanner(os.Stdin)
    sc.Buffer(make([]byte, 1<<20), 1<<20)
    sc.Scan()
    t, _ := strconv.Atoi(strings.TrimSpace(sc.Text()))
    for i := 0; i < t; i++ {
        sc.Scan()
        w := strings.Fields(sc.Text())
        for a, b := 0, len(w)-1; a < b; a, b = a+1, b-1 {
            w[a], w[b] = w[b], w[a]
        }
        fmt.Println(strings.Join(w, " "))
    }
}
''',
        "recursion": r'''package main

import "fmt"

func f(n int) int64 {
    if n == 0 {
        return 0
    }
    r := f(n - 1)
    return r + int64(n&1)
}

func main() {
    var n int
    fmt.Scan(&n)
    fmt.Println(f(n))
}
''',
        "wrong": 'package main\n\nimport "fmt"\n\nfunc main() { fmt.Println(42) }\n',
        "loop": "package main\n\nfunc main() {\n\tx := 0\n\tfor {\n\t\tx++\n\t}\n}\n",
        "syntax": "package main\n\nfunc main( {\n}\n",
        "crash": 'package main\n\nfunc main() { panic("boom") }\n',
        "isolation": r'''package main

import (
    "fmt"
    "os"
)

func main() {
    escaped := false
    if _, err := os.ReadFile(`READ`); err == nil {
        escaped = true
    }
    if err := os.WriteFile(`WRITE`, []byte{1}, 0644); err == nil {
        escaped = true
    }
    if escaped {
        fmt.Println("ESCAPED")
    } else {
        fmt.Println("DENIED")
    }
}
''',
    },
    "rust": {
        "sum64": r'''use std::io::{self, Read, Write};

fn main() {
    let mut s = String::new();
    io::stdin().read_to_string(&mut s).unwrap();
    let mut it = s.split_ascii_whitespace();
    let n: usize = it.next().unwrap().parse().unwrap();
    let mut t: i64 = 0;
    for _ in 0..n {
        t += it.next().unwrap().parse::<i64>().unwrap();
    }
    let out = io::stdout();
    let mut o = out.lock();
    writeln!(o, "{}", t).unwrap();
}
''',
        "lines": r'''use std::io::{self, BufRead};

fn main() {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let t: usize = lines.next().unwrap().unwrap().trim().parse().unwrap();
    for _ in 0..t {
        let l = lines.next().unwrap().unwrap();
        let mut w: Vec<&str> = l.split_whitespace().collect();
        w.reverse();
        println!("{}", w.join(" "));
    }
}
''',
        "recursion": r'''use std::io::Read;

fn f(n: u64) -> u64 {
    let mut pad = [0u8; 64];
    unsafe { std::ptr::write_volatile(&mut pad[(n & 63) as usize], n as u8) };
    if n == 0 {
        return 0;
    }
    let r = f(n - 1);
    r + (unsafe { std::ptr::read_volatile(&pad[(n & 63) as usize]) } & 1) as u64
}

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    println!("{}", f(s.trim().parse().unwrap()));
}
''',
        "wrong": 'fn main() { println!("42"); }\n',
        "loop": "fn main() {\n    let mut x: u64 = 0;\n    loop {\n        x = std::hint::black_box(x + 1);\n    }\n}\n",
        "syntax": "fn main( {}\n",
        "crash": 'fn main() {\n    let i: usize = "3".parse().unwrap();\n    let v = vec![1, 2];\n    println!("{}", v[i]);\n}\n',
        "isolation": r'''fn main() {
    let mut escaped = false;
    if std::fs::read(r"READ").is_ok() {
        escaped = true;
    }
    if std::fs::write(r"WRITE", [1u8]).is_ok() {
        escaped = true;
    }
    println!("{}", if escaped { "ESCAPED" } else { "DENIED" });
}
''',
    },
    "javascript": {
        "sum64": r'''const data = require('fs').readFileSync(0, 'utf8').split(/\s+/).filter(Boolean);
const n = Number(data[0]);
let s = 0n;
for (let i = 1; i <= n; i++) s += BigInt(data[i]);
console.log(s.toString());
''',
        "lines": r'''const rl = require('readline').createInterface({ input: process.stdin });
const lines = [];
rl.on('line', (l) => lines.push(l));
rl.on('close', () => {
  const t = parseInt(lines[0], 10);
  const out = [];
  for (let i = 1; i <= t; i++) out.push(lines[i].trim().split(/\s+/).reverse().join(' '));
  console.log(out.join('\n'));
});
''',
        "recursion": r'''import { readFileSync } from 'fs';

const n = Number(readFileSync('/dev/stdin', 'utf8').trim());
function f(k) {
  if (k === 0) return 0;
  const r = f(k - 1);
  return r + (k & 1);
}
console.log(f(n));
''',
        "wrong": "console.log(42);\n",
        "loop": "let x = 0;\nfor (;;) { x++; }\n",
        "syntax": "const x = ;\n",
        "crash": "throw new Error('boom');\n",
        "isolation": r'''const fs = require('fs');
let escaped = false;
try { fs.readFileSync('READ'); escaped = true; } catch (e) { }
try { fs.writeFileSync('WRITE', 'x'); escaped = true; } catch (e) { }
console.log(escaped ? 'ESCAPED' : 'DENIED');
''',
    },
    "typescript": {
        "sum64": r'''import * as fs from 'fs';

const data: string[] = fs.readFileSync(0, 'utf8').split(/\s+/).filter((x: string) => x.length > 0);
const n: number = Number(data[0]);
let s: bigint = 0n;
for (let i = 1; i <= n; i++) s += BigInt(data[i]);
console.log(s.toString());
''',
        "lines": r'''interface Line {
  words: string[];
}
enum Mode {
  Reverse = 1,
}
declare const require: any;
const rows: string[] = require('fs').readFileSync(0, 'utf8').split('\n');
const t: number = parseInt(rows[0], 10);
const out: string[] = [];
for (let i = 1; i <= t; i++) {
  const l: Line = { words: rows[i].trim().split(/\s+/) };
  if (Mode.Reverse) l.words.reverse();
  out.push(l.words.join(' '));
}
console.log(out.join('\n'));
''',
        "recursion": r'''const n: number = Number(require('fs').readFileSync(0, 'utf8').trim());
function f(k: number): number {
  if (k === 0) return 0;
  const r: number = f(k - 1);
  return r + (k & 1);
}
console.log(f(n));
''',
        "wrong": "console.log(42);\n",
        "loop": "let x: number = 0;\nfor (;;) { x++; }\n",
        "syntax": "let x: number = ;\n",
        "crash": "throw new Error('boom');\n",
        "isolation": r'''import * as fs from 'fs';
let escaped: boolean = false;
try { fs.readFileSync('READ'); escaped = true; } catch (e) { }
try { fs.writeFileSync('WRITE', 'x'); escaped = true; } catch (e) { }
console.log(escaped ? 'ESCAPED' : 'DENIED');
''',
    },
}

EXPECT = {"sum64": "ok", "lines": "ok", "recursion": "ok", "online_judge": "ok", "wrong": "wrong_answer",
          "loop": "timeout", "syntax": "compile_error", "crash": "runtime_error", "isolation": "ok"}


def programs(lang, windows=None):
    """[{name, code, cases, expect, timeout}] for one language. The isolation program only where it means something
    (Windows, inside the sandbox)."""
    windows = polyglot.WINDOWS if windows is None else windows
    out = []
    for name, code in PROGRAMS[lang].items():
        if name == "isolation":
            if not windows:
                continue
            read, write = _paths(lang)
            code = code.replace("READ", read).replace("WRITE", write)
        cases = {"sum64": SUM_CASES, "lines": LINES_CASES, "recursion": depth_cases(lang), "online_judge": PAIR_CASES,
                 "isolation": [("", "DENIED\n")]}.get(name, [("3\n1 2 3\n", "6\n")])
        expect = EXPECT[name]
        if name == "syntax" and lang == "javascript":
            expect = "runtime_error"  # no build step: Node reports the syntax error when it runs the file
        out.append({"name": name, "code": code, "cases": cases, "expect": expect,
                    "timeout": 3.0 if name == "loop" else 20.0})
    return out
