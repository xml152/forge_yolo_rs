"""支持 ``python -m engine <子命令> ...``。"""

from . import main

raise SystemExit(main())
