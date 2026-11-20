// Interactive setup assistant for Timbal Connect.
//
// The reader picks an OS, a data source and a few options; the component then builds the exact
// commands and configuration to run, in order. Everything runs in the browser: nothing is sent
// anywhere, and passwords are never asked for (the output uses the placeholder CHANGE_ME).
//
// Colours are theme-neutral (inherited text, translucent greys) so it works in light and dark mode.
//
// Mintlify compiles each exported component on its own, so everything it needs (data, helpers, UI
// functions) lives inside this one component. `__expose` is only used by tests.

export const ConnectGuidedSetup = ({ __expose } = {}) => {
  const PLACEHOLDER = "CHANGE_ME";

  const OS_LIST = [
    { id: "windows", label: "Windows" },
    { id: "linux", label: "Linux" },
    { id: "macos", label: "macOS" },
  ];

  const SOURCES = {
    postgres: { label: "PostgreSQL", kind: "postgres", port: "5432", host: "127.0.0.1", db: "appdb", user: "timbal_ro", name: "main" },
    mysql: { label: "MySQL", kind: "mysql", port: "3306", host: "127.0.0.1", db: "shopdb", user: "timbal_ro", name: "shop" },
    mariadb: { label: "MariaDB", kind: "mysql", port: "3306", host: "127.0.0.1", db: "shopdb", user: "timbal_ro", name: "shop" },
    sqlserver: { label: "SQL Server", kind: "sql_server", port: "1433", host: "sql01.corp.local", db: "ERP", user: "timbal_ro", name: "erp" },
    none: { label: "Files only", kind: null, port: "", host: "", db: "", user: "", name: "main" },
  };

  const FS_PATHS = {
    windows: "D:\\data\\exports",
    linux: "/srv/exports",
    macos: "/Users/Shared/exports",
  };

  const INITIAL_STATE = {
    os: "windows",
    source: "sqlserver",
    withFiles: false,
    layout: "file", // "file" = connections.json, "inline" = inside config.json
    reach: "default", // SQL Server: default | named_port | named_browser
    auth: "integrated", // SQL Server on Windows: sql | integrated | windows
    encrypt: "trust", // SQL Server: trust | verify | none
    query: "read", // none | read | write
    o: {}, // free-text overrides typed by the reader
  };

  // ---------------------------------------------------------------------------
  // Resolving the reader's choices into concrete values
  // ---------------------------------------------------------------------------

  const resolve = (s) => {
    const src = SOURCES[s.source];
    const o = s.o || {};
    const pick = (key, fallback) => (o[key] !== undefined && String(o[key]).trim() !== "" ? String(o[key]).trim() : fallback);
    const isSqlServer = s.source === "sqlserver";
    const defaultPort = isSqlServer && s.reach === "named_port" ? "52001" : src.port;
    return {
      os: s.os,
      source: s.source,
      src,
      isDb: s.source !== "none",
      isSqlServer,
      withFiles: s.source === "none" ? true : s.withFiles,
      layout: s.source === "none" ? "inline" : s.layout,
      reach: s.reach,
      // Integrated and Windows authentication only exist on Windows connectors.
      auth: isSqlServer && s.os === "windows" ? s.auth : "sql",
      encrypt: s.encrypt,
      query: s.query,
      name: pick("name", src.name),
      host: pick("host", src.host),
      port: pick("port", defaultPort),
      db: pick("db", src.db),
      user: pick("user", src.user),
      instance: pick("instance", "INSTANCE"),
      svcAccount: pick("svcAccount", "CORP\\svc_timbal$"),
      ntlmUser: pick("ntlmUser", "CORP\\svc_timbal"),
      fsName: pick("fsName", "exports"),
      fsPath: pick("fsPath", FS_PATHS[s.os]),
      proxy: pick("proxy", ""),
    };
  };

  // ---------------------------------------------------------------------------
  // Config builders
  // ---------------------------------------------------------------------------

  const buildEntry = (c) => {
    const user = encodeURIComponent(c.user);
    if (c.source === "postgres") {
      return { kind: "postgres", dsn: `postgres://${user}:${PLACEHOLDER}@${c.host}:${c.port}/${c.db}`, query: c.query };
    }
    if (c.source === "mysql" || c.source === "mariadb") {
      return { kind: "mysql", dsn: `mysql://${user}:${PLACEHOLDER}@${c.host}:${c.port}/${c.db}`, query: c.query };
    }
    // SQL Server
    const server = c.reach === "named_browser" ? `tcp:${c.host}\\${c.instance}` : `tcp:${c.host},${c.port}`;
    const parts = [`Server=${server}`, `Database=${c.db}`];
    if (c.auth === "sql") {
      parts.push(`User Id=${c.user}`, `Password=${PLACEHOLDER}`);
    }
    if (c.encrypt === "trust") parts.push("Encrypt=true", "TrustServerCertificate=true");
    if (c.encrypt === "verify") parts.push("Encrypt=true", "TrustServerCertificate=false");
    if (c.encrypt === "none") parts.push("Encrypt=false");
    const entry = { kind: "sql_server", dsn: parts.join(";") };
    if (c.auth === "integrated") entry.auth = { mode: "integrated" };
    if (c.auth === "windows") entry.auth = { mode: "windows", user: c.ntlmUser, password: PLACEHOLDER };
    entry.query = c.query;
    return entry;
  };

  const json = (value) => JSON.stringify(value, null, 2);

  const buildFiles = (c) => {
    // Returns the files to write: connections.json (a bare map) and/or the keys to put in config.json.
    const files = { connections: null, config: null };
    const fs = c.withFiles ? { [c.fsName]: { path: c.fsPath } } : null;
    if (c.isDb && c.layout === "file") {
      files.connections = { [c.name]: buildEntry(c) };
      if (fs) files.config = { filesystems: fs };
    } else {
      const cfg = {};
      if (c.isDb) cfg.connections = { [c.name]: buildEntry(c) };
      if (fs) cfg.filesystems = fs;
      files.config = cfg;
    }
    return files;
  };

  // ---------------------------------------------------------------------------
  // OS-specific command helpers
  // ---------------------------------------------------------------------------

  const winDir = "C:\\ProgramData\\TimbalConnect";
  const unixDir = "/etc/timbal-connect";

  const filePath = (os, name) => (os === "windows" ? `${winDir}\\${name}` : `${unixDir}/${name}`);

  const runBinary = (os, args) => {
    if (os === "windows") return `& "$env:ProgramFiles\\TimbalConnect\\current\\timbal-connect.exe" ${args}`;
    if (os === "linux") {
      return `sudo bash -c 'set -a; . /etc/timbal-connect/connector.env; exec /opt/timbal-connect/current/timbal-connect ${args}'`;
    }
    return `sudo /opt/timbal-connect/current/timbal-connect ${args}`;
  };

  const restartCmd = (os) => {
    if (os === "windows") return "Restart-Service timbal-connect";
    if (os === "linux") return "sudo systemctl restart timbal-connect";
    return "sudo launchctl kickstart -k system/ai.timbal.connect";
  };

  const logCmd = (os) => {
    if (os === "windows") return "Get-Content -Wait C:\\ProgramData\\TimbalConnect\\timbal-connect.log";
    if (os === "linux") return "journalctl -fu timbal-connect";
    return "tail -f /Library/Logs/timbal-connect.log";
  };

  const createFileCmd = (os, name, content) => {
    if (os === "windows") {
      return [
        `$p = "$env:ProgramData\\TimbalConnect\\${name}"`,
        "$json = @'",
        content,
        "'@",
        "[System.IO.File]::WriteAllText($p, $json)   # UTF-8 without BOM",
      ].join("\n");
    }
    return [`sudo tee ${unixDir}/${name} > /dev/null <<'EOF'`, content, "EOF"].join("\n");
  };

  const mergeConfigCmd = (os, content) => {
    if (os === "windows") {
      return [
        '$p = "$env:ProgramData\\TimbalConnect\\config.json"',
        "$add = @'",
        content,
        "'@ | ConvertFrom-Json",
        "$cfg = Get-Content $p -Raw | ConvertFrom-Json",
        "foreach ($k in $add.PSObject.Properties.Name) {",
        "  $cfg | Add-Member -NotePropertyName $k -NotePropertyValue $add.$k -Force",
        "}",
        "[System.IO.File]::WriteAllText($p, ($cfg | ConvertTo-Json -Depth 10))   # UTF-8 without BOM",
      ].join("\n");
    }
    if (os === "macos") {
      return [
        "# Needs jq (included in recent macOS; otherwise: brew install jq)",
        "cat > /tmp/timbal-add.json <<'EOF'",
        content,
        "EOF",
        `sudo jq -s '.[0] * .[1]' ${unixDir}/config.json /tmp/timbal-add.json > /tmp/timbal-merged.json`,
        `sudo tee ${unixDir}/config.json < /tmp/timbal-merged.json > /dev/null`,
        "rm /tmp/timbal-add.json /tmp/timbal-merged.json",
      ].join("\n");
    }
    return createFileCmd(os, "config.json", content);
  };

  const reachCmd = (os, host, port) => {
    if (os === "windows") return `Test-NetConnection ${host} -Port ${port}   # look for TcpTestSucceeded : True`;
    return `nc -vz -w 5 ${host} ${port}`;
  };

  // ---------------------------------------------------------------------------
  // SQL to create a read-only login
  // ---------------------------------------------------------------------------

  const loginSql = (c) => {
    if (c.source === "postgres") {
      return [
        `-- psql -h ${c.host} -p ${c.port} -U postgres -d ${c.db}`,
        `CREATE ROLE ${c.user} LOGIN PASSWORD '${PLACEHOLDER}';`,
        `GRANT CONNECT ON DATABASE ${c.db} TO ${c.user};`,
        `GRANT USAGE ON SCHEMA public TO ${c.user};`,
        `GRANT SELECT ON ALL TABLES IN SCHEMA public TO ${c.user};`,
        `ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ${c.user};`,
      ].join("\n");
    }
    if (c.source === "mysql" || c.source === "mariadb") {
      const client = c.source === "mariadb" ? "mariadb" : "mysql";
      return [
        `-- ${client} -h ${c.host} -P ${c.port} -u root -p`,
        `-- '%' accepts any client address: replace it with the connector server's address if you can.`,
        `CREATE USER '${c.user}'@'%' IDENTIFIED BY '${PLACEHOLDER}';`,
        `GRANT SELECT ON ${c.db}.* TO '${c.user}'@'%';`,
      ].join("\n");
    }
    if (c.auth === "sql") {
      return [
        `-- sqlcmd -S ${c.reach === "named_browser" ? `${c.host}\\${c.instance}` : `${c.host},${c.port}`} -U sa -C   (or SQL Server Management Studio)`,
        `CREATE LOGIN [${c.user}] WITH PASSWORD = '${PLACEHOLDER}';`,
        `USE [${c.db}];`,
        `CREATE USER [${c.user}] FOR LOGIN [${c.user}];`,
        `ALTER ROLE db_datareader ADD MEMBER [${c.user}];`,
      ].join("\n");
    }
    const account = c.auth === "integrated" ? c.svcAccount : c.ntlmUser;
    return [
      `CREATE LOGIN [${account}] FROM WINDOWS;`,
      `USE [${c.db}];`,
      `CREATE USER [${account}] FOR LOGIN [${account}];`,
      `ALTER ROLE db_datareader ADD MEMBER [${account}];`,
    ].join("\n");
  };

  // ---------------------------------------------------------------------------
  // The plan: ordered steps with the exact things to run
  // ---------------------------------------------------------------------------

  const buildPlan = (c) => {
    const steps = [];
    const files = buildFiles(c);
    const where = { workstation: "Your workstation", server: "The Timbal Connect server", database: "The database" };

    // 1. Enrollment token ------------------------------------------------------------------------
    const tokenBlocks = [
      {
        label: "bash",
        code: [
          `curl -X POST "https://api.timbal.ai/orgs/$ORG_ID/connectors/enrollment-tokens" \\`,
          `  -H "Authorization: Bearer $API_KEY" \\`,
          `  -H "Content-Type: application/json" \\`,
          `  -d '{"name": "${c.isDb ? c.name : c.fsName} server", "ttl_secs": 86400}'`,
        ].join("\n"),
      },
    ];
    if (c.os === "windows") {
      tokenBlocks.push({
        label: "PowerShell",
        code: [
          "Invoke-RestMethod -Method Post `",
          '  -Uri "https://api.timbal.ai/orgs/$env:ORG_ID/connectors/enrollment-tokens" `',
          '  -Headers @{ Authorization = "Bearer $env:API_KEY" } `',
          '  -ContentType "application/json" `',
          `  -Body '{"name": "${c.isDb ? c.name : c.fsName} server", "ttl_secs": 86400}'`,
        ].join("\n"),
      });
    }
    steps.push({
      id: "token",
      title: "Create an enrollment token",
      where: where.workstation,
      text: "Run this anywhere you have an API key and your organization id. The response contains `token`, shown only once. It is single-use and expires after 24 hours (`ttl_secs` goes up to 30 days).",
      blocks: tokenBlocks,
    });

    // 2. Install -----------------------------------------------------------------------------------
    if (c.os === "windows") {
      const proxy = c.proxy ? ` -Proxy ${c.proxy}` : "";
      steps.push({
        id: "install",
        title: "Install the connector",
        where: where.server,
        text: "In an **elevated** command prompt or PowerShell on the server. If you prefer a window, download the file and double-click it instead: [Install on Windows](/connect/install-windows).",
        blocks: [
          {
            label: "Download",
            code: "curl.exe -fLo TimbalConnectSetup.cmd https://api.timbal.ai/v1/connect/releases/TimbalConnectSetup.cmd",
          },
          { label: "Install", code: `.\\TimbalConnectSetup.cmd -Token <TOKEN>${proxy}` },
        ],
        note: "`curl.exe` is not on Windows Server 2016: download the file from the browser instead. If the installer says the file is incomplete, download it again; do not edit it.",
      });
    } else {
      const proxy = c.proxy ? ` \\\n  --proxy ${c.proxy}` : "";
      steps.push({
        id: "install",
        title: "Install the connector",
        where: where.server,
        text: "On the server, as a user with `sudo`. The script detects the OS and architecture, checks the download's SHA-256 and starts the service.",
        blocks: [
          {
            label: "bash",
            code: `curl -fsSL https://api.timbal.ai/v1/connect/releases/install.sh | sudo bash -s -- \\\n  --token <TOKEN>${proxy}`,
          },
        ],
      });
    }

    // 3. Prepare the database ---------------------------------------------------------------------
    if (c.isDb) {
      const label = c.src.label;
      if (c.reach === "named_browser" && c.isSqlServer) {
        steps.push({
          id: "reach",
          title: "Check the server can reach the SQL Server instance",
          where: where.server,
          text: `A named instance without a port (\`${c.host}\\${c.instance}\`) is found through the SQL Browser service: the server needs **UDP 1434** to \`${c.host}\`, and TCP to the instance's dynamic port. If either is blocked, switch the "How to reach it" option to an explicit port.`,
          blocks: [],
        });
      } else {
        steps.push({
          id: "reach",
          title: `Check the server can reach ${label}`,
          where: where.server,
          text: `The connector connects from this server to \`${c.host}:${c.port}\`. Check it before writing any configuration.`,
          blocks: [{ label: c.os === "windows" ? "PowerShell" : "bash", code: reachCmd(c.os, c.host, c.port) }],
        });
      }

      if (c.auth === "integrated") {
        const gmsa = c.svcAccount.endsWith("$");
        steps.push({
          id: "service-account",
          title: "Run the service as a domain account",
          where: where.server,
          text: `Integrated authentication uses the account the service runs as. The default, \`NT AUTHORITY\\NetworkService\`, reaches another machine as the computer account, which rarely has a SQL login. ${
            gmsa
              ? "A group managed service account (the name ends with `$`) needs no password, so the password is left empty."
              : "For a normal domain user, replace `<PASSWORD>` with its password."
          }`,
          blocks: [
            {
              label: "PowerShell (elevated)",
              code: `sc.exe config timbal-connect obj= "${c.svcAccount}" password= "${gmsa ? "" : "<PASSWORD>"}"\nRestart-Service timbal-connect`,
            },
          ],
          note: `If SQL Server runs on this same machine, \`NetworkService\` often works without changing the account. Skip this step and see whether \`doctor\` passes.`,
        });
      }

      const loginNote = [
        c.auth === "sql" && c.isSqlServer
          ? "This needs SQL Server to allow SQL logins (mixed mode). Many Business Central and NAV servers only allow Windows logins: pick Windows or integrated authentication above."
          : "",
        c.query === "write"
          ? "This login is read-only. You chose to allow writes from Timbal, so also grant the write privileges you want (for example `INSERT`, `UPDATE`), and only on the tables that need them."
          : "A read-only login limits the damage if anything is ever misused. The `query` setting is a second guard, not a replacement.",
      ]
        .filter(Boolean)
        .join(" ");
      steps.push({
        id: "login",
        title: c.auth === "sql" ? "Create a read-only login" : "Give the account read access",
        where: where.database,
        text:
          c.auth === "sql"
            ? `Run this as a ${label} administrator. Replace \`${PLACEHOLDER}\` with a strong password.`
            : "Run this as a SQL Server administrator. No password is involved: SQL Server trusts the Windows account.",
        blocks: [{ label: "SQL", code: loginSql(c) }],
        note: loginNote,
      });
    }

    // 4. Write the configuration -----------------------------------------------------------------------
    const configBlocks = [];
    const secretFile = c.isDb && (c.auth === "sql" || c.auth === "windows") ? (files.connections ? "connections.json" : "config.json") : null;

    if (files.connections) {
      configBlocks.push({
        label: `Create ${filePath(c.os, "connections.json")}${c.os === "windows" ? " (PowerShell, elevated)" : ""}`,
        code: createFileCmd(c.os, "connections.json", json(files.connections)),
      });
    }
    if (files.config) {
      if (c.os === "linux") {
        configBlocks.push({
          label: `Create ${filePath(c.os, "config.json")} (it does not exist yet on Linux)`,
          code: mergeConfigCmd(c.os, json(files.config)),
        });
      } else {
        configBlocks.push({
          label: `Add to ${filePath(c.os, "config.json")}, keeping its other keys`,
          code: mergeConfigCmd(c.os, json(files.config)),
        });
      }
    }

    const configText = [];
    if (c.isDb && files.connections) {
      configText.push(
        `Each entry is named: \`${c.name}\` is the name Timbal will see. The connection string stays on this server.`
      );
    }
    if (c.isDb && !files.connections) {
      configText.push(
        `Connections go inside \`config.json\` under \`connections\` (version 0.1.27 or newer). \`${c.name}\` is the name Timbal will see; the connection string stays on this server.`
      );
    }
    if (c.withFiles) {
      configText.push(
        `File shares always go in \`config.json\` under \`filesystems\`. \`${c.fsName}\` is the name Timbal will see; \`${c.fsPath}\` stays on this server.`
      );
    }
    const configNotes = [];
    if (files.connections) configNotes.push("This replaces `connections.json`. If it already has connections, add the entry to the existing file instead.");
    if (c.os === "windows") configNotes.push("Save files as UTF-8 **without** BOM. The commands above do. A byte-order mark makes the file fail to parse.");
    if (files.config) configNotes.push("Keep `connections_path` out of `config.json` when you use inline `connections`: setting both stops the connector from starting.");
    if (files.connections) configNotes.push("An inline `connections` block already present in `config.json` takes precedence over `connections.json`.");
    if (c.os === "linux" && files.config) {
      configNotes.push("If `config.json` already exists on this server, add the keys to it by hand instead of creating it.");
      configNotes.push("On Linux the connector settings live in `connector.env`. If `doctor` does not show your entries, add `TIMBAL_CONNECT_CONFIG=/etc/timbal-connect/config.json` to that file and restart.");
    }

    steps.push({
      id: "config",
      title: "Write the configuration",
      where: where.server,
      text: configText.join(" "),
      blocks: configBlocks,
      note: configNotes.join(" "),
    });

    // 5. Fill in the password --------------------------------------------------------------------------
    if (secretFile) {
      const target = filePath(c.os, secretFile);
      steps.push({
        id: "secret",
        title: `Replace ${PLACEHOLDER} with the real password`,
        where: where.server,
        text: `Open \`${target}\` and replace \`${PLACEHOLDER}\` with the password. ${c.isSqlServer ? (c.auth === "windows" ? "The password is stored as written, with no connection-string quoting." : "In the connection string, wrap the password in double quotes if it contains a semicolon.") : "In the URL connection string, percent-encode special characters (`@` becomes `%40`)."} In JSON, write \`\\\\\` for a backslash and \`\\"\` for a quote.`,
        blocks: [
          {
            label: c.os === "windows" ? "PowerShell (elevated)" : "bash",
            code: c.os === "windows" ? `notepad "${target}"` : `sudo nano ${target}`,
          },
        ],
        note:
          c.os === "windows"
            ? "Do not save as \"UTF-8 with BOM\": the file would fail to parse."
            : c.os === "linux"
            ? "This file holds a password. Make sure only root and the service account can read it; find the service user with `systemctl show timbal-connect -p User`."
            : "This file holds a password. Make sure only root can read it.",
      });
    }

    // 6. Restart -------------------------------------------------------------------------------------------
    steps.push({
      id: "restart",
      title: "Restart the service",
      where: where.server,
      text: "The connector reads its configuration at startup, so restart it after every change.",
      blocks: [{ label: c.os === "windows" ? "PowerShell (elevated)" : "bash", code: restartCmd(c.os) }],
    });

    // 7. Verify --------------------------------------------------------------------------------------------
    const verifyBlocks = [{ label: "Full check", code: runBinary(c.os, "doctor") }];
    if (c.isDb) {
      verifyBlocks.push({ label: `Only ${c.name}`, code: runBinary(c.os, `doctor --connection ${c.name}`) });
      const sampleTable = c.source === "sqlserver" ? "dbo.<table>" : c.source === "postgres" ? "public.<table>" : `${c.db}.<table>`;
      verifyBlocks.push({ label: "A table's columns, primary key and cursor", code: runBinary(c.os, `doctor --connection ${c.name} --table "${sampleTable}"`) });
      verifyBlocks.push({ label: "A real query", code: runBinary(c.os, `query --connection ${c.name} "SELECT 1"`) });
    }
    verifyBlocks.push({ label: "Watch the log", code: logCmd(c.os) });
    steps.push({
      id: "verify",
      title: "Verify on the server",
      where: where.server,
      text: `\`doctor\` is read-only. \`[config]\` must parse and \`[egress]\` must reach Timbal on port 443${
        c.isDb ? `. \`[connection: ${c.name}]\` must connect and list tables` : ""
      }${c.withFiles ? `. The \`${c.fsName}\` share must be readable` : ""}. In the log you want \`connector enrolled and connected\`.`,
      blocks: verifyBlocks,
      note:
        c.os === "windows" && c.isSqlServer && c.auth === "integrated"
          ? "`doctor` from your own PowerShell logs in as you, not as the service account. The real test is the `[connection]` result in the log after the service restart."
          : "",
    });

    // 8. Verify from Timbal ----------------------------------------------------------------------------------
    const timbal = [
      {
        label: "Find the connector id",
        code: `curl "https://api.timbal.ai/orgs/$ORG_ID/connectors" \\\n  -H "Authorization: Bearer $API_KEY"`,
      },
    ];
    if (c.isDb) {
      timbal.push({
        label: "List the connections",
        code: `curl "https://api.timbal.ai/orgs/$ORG_ID/connectors/$CONNECTOR_ID/browse/connections" \\\n  -H "Authorization: Bearer $API_KEY"\n# -> { "connections": ["${c.name}"] }`,
      });
      timbal.push({
        label: "List the tables",
        code: `curl "https://api.timbal.ai/orgs/$ORG_ID/connectors/$CONNECTOR_ID/browse/tables?connection_ref=${c.name}" \\\n  -H "Authorization: Bearer $API_KEY"`,
      });
    }
    if (c.withFiles) {
      timbal.push({
        label: "List the file shares",
        code: `curl "https://api.timbal.ai/orgs/$ORG_ID/connectors/$CONNECTOR_ID/browse/filesystems" \\\n  -H "Authorization: Bearer $API_KEY"\n# -> { "filesystems": ["${c.fsName}"] }`,
      });
      timbal.push({
        label: "Summarize a folder before copying it",
        code: `curl "https://api.timbal.ai/orgs/$ORG_ID/connectors/$CONNECTOR_ID/browse/filesystems/summary?filesystem_ref=${c.fsName}" \\\n  -H "Authorization: Bearer $API_KEY"`,
      });
    }
    steps.push({
      id: "timbal",
      title: "Confirm from Timbal",
      where: where.workstation,
      text: "The connector must be online. Your names appear exactly as you wrote them, and nothing else about the server does.",
      blocks: timbal,
      note: "Next: [sync the data](/connect/sync-data)" + (c.withFiles ? " or [copy the files](/connect/filesystems#copy-files-to-a-knowledge-base)" : "") + ", or [connect your IDE](/connect/mcp).",
    });

    return steps;
  };

  const buildTroubleshooting = (c) => {
    const items = [
      ["The connector shows as offline", "Run `doctor` and read `[egress]`. See [Network requirements](/connect/network-requirements). A proxy or TLS inspection is the usual cause."],
      ["`invalid or expired enrollment token` in the log", "The token expired, was used already, or is from another environment. Create a new one and run the installer again with it: an existing install re-enrolls."],
    ];
    if (c.isDb) {
      items.push(["`doctor` can't parse the file", "A byte-order mark, or an unescaped backslash in a JSON string. Save as UTF-8 without BOM and write `\\\\` for each `\\`."]);
    }
    if (c.isSqlServer && c.auth === "sql") {
      items.push(["`Login failed for user`", "The login does not exist, or the server only allows Windows logins (mixed mode off). Use Windows or integrated authentication."]);
    }
    if (c.isSqlServer && c.auth === "integrated") {
      items.push(["`Login failed for user 'CORP\\SERVERNAME$'`", "The service still runs as `NetworkService`. Run it as the domain account and create its login (steps above)."]);
    }
    if (c.isSqlServer && c.auth === "windows") {
      items.push(["`Login failed` with NTLM", "`user` must be `DOMAIN\\user` (in JSON, `DOMAIN\\\\user`). If the domain blocks NTLM, use integrated authentication with a service account."]);
    }
    if (c.isSqlServer && c.reach === "named_browser") {
      items.push(["The named instance never connects", "SQL Browser is off or UDP 1434 is blocked. Use `Server=tcp:host,PORT` with the instance's static port."]);
    }
    if (c.isSqlServer && c.encrypt === "verify") {
      items.push(["TLS error: the certificate chain is not trusted", "Install the CA on this server, make `Server=` match the certificate's name, or use `TrustServerCertificate=true` for a self-signed certificate."]);
    }
    if (c.isDb || c.withFiles) {
      items.push(["`doctor` mentions a remote local-config override", "A map pushed from Timbal replaces this one until you run `timbal-connect local-config clear` and restart."]);
    }
    if (c.withFiles) {
      items.push(["The share is listed but not readable", c.os === "windows" ? "The service runs as `NT AUTHORITY\\NetworkService`: give it read access to the folder, or run the service as an account that has it." : c.os === "linux" ? "The service runs as the `timbal-connect` user and the unit sets `ProtectHome=true`, so anything under `/home` is hidden. Use another path and give that user read access." : "Check the folder is readable by the service (it runs as root under launchd) and not blocked by macOS privacy settings."]);
    }
    return items;
  };

  if (__expose) __expose({ resolve, buildPlan, buildTroubleshooting, INITIAL_STATE, SOURCES });

  // -------------------------------------------------------------------------
  // UI helpers. They are plain functions that return JSX, not components: a component
  // declared inside another one is a new type on every render and would reset the inputs.
  // -------------------------------------------------------------------------

  const [s, setS] = useState(INITIAL_STATE);
  const [copiedKey, setCopiedKey] = useState(null);
  const [dark, setDark] = useState(false);

  // Follow the site's light/dark switch (Mintlify toggles a `dark` class on <html>).
  useEffect(() => {
    const root = document.documentElement;
    const read = () => setDark(root.classList.contains("dark"));
    read();
    const observer = new MutationObserver(read);
    observer.observe(root, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);

  // Palette taken from the docs theme (Mintlify "mint", monochrome): near-black cards with hairline
  // borders in dark mode, white cards in light mode. The selected option is shown by a stronger
  // border and a soft tint, not by inverting the colours.
  const t = dark
    ? {
        panel: "#0e0e10", panelBorder: "rgba(255, 255, 255, 0.12)", field: "#000000", fieldBorder: "rgba(255, 255, 255, 0.22)",
        text: "#f2f2f2", muted: "#a1a1aa", placeholder: "#80808a", chip: "transparent", chipBorder: "rgba(255, 255, 255, 0.16)",
        activeBg: "rgba(255, 255, 255, 0.10)", activeBorder: "rgba(255, 255, 255, 0.7)", activeText: "#ffffff",
        inlineCode: "rgba(255, 255, 255, 0.10)",
        codeHead: "#161619", codeBody: "#0a0a0c", codeBorder: "rgba(255, 255, 255, 0.12)", codeText: "#e8e8ea", codeMuted: "#a1a1aa",
        copyBorder: "rgba(255, 255, 255, 0.22)", ok: "#7ee787",
      }
    : {
        panel: "#ffffff", panelBorder: "rgba(10, 10, 10, 0.12)", field: "#ffffff", fieldBorder: "rgba(10, 10, 10, 0.25)",
        text: "#3e3e3e", muted: "#6b6b73", placeholder: "#a1a1aa", chip: "transparent", chipBorder: "rgba(10, 10, 10, 0.18)",
        activeBg: "rgba(10, 10, 10, 0.06)", activeBorder: "#0a0a0a", activeText: "#0a0a0a",
        inlineCode: "rgba(10, 10, 10, 0.07)",
        codeHead: "#18181b", codeBody: "#0e0e10", codeBorder: "rgba(10, 10, 10, 0.12)", codeText: "#e8e8ea", codeMuted: "#a1a1aa",
        copyBorder: "rgba(255, 255, 255, 0.25)", ok: "#7ee787",
      };
  const line = `1px solid ${t.panelBorder}`;

  const rich = (text) => {
    // Minimal inline markup: `code`, **bold** and [label](href).
    const out = [];
    const re = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;
    let last = 0;
    let m;
    let i = 0;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) out.push(text.slice(last, m.index));
      const tok = m[0];
      if (tok.startsWith("`")) {
        out.push(
          <code key={i++} style={{ background: t.inlineCode, color: t.text, borderRadius: 4, padding: "1px 5px", fontSize: "0.88em" }}>
            {tok.slice(1, -1)}
          </code>
        );
      } else if (tok.startsWith("**")) {
        out.push(<strong key={i++}>{tok.slice(2, -2)}</strong>);
      } else {
        const parts = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok);
        out.push(
          <a key={i++} href={parts[2]} style={{ textDecoration: "underline" }}>
            {parts[1]}
          </a>
        );
      }
      last = m.index + tok.length;
    }
    if (last < text.length) out.push(text.slice(last));
    return out;
  };

  const copy = (key, code) => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(code).then(() => {
        setCopiedKey(key);
        setTimeout(() => setCopiedKey((k) => (k === key ? null : k)), 1500);
      });
    }
  };

  const codeBlock = (key, label, code) => (
    <div key={key} style={{ margin: "10px 0", borderRadius: 10, overflow: "hidden", border: `1px solid ${t.codeBorder}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", background: t.codeHead, color: t.codeMuted, fontSize: 12, padding: "6px 10px" }}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
        <button
          type="button"
          onClick={() => copy(key, code)}
          style={{ background: "transparent", color: copiedKey === key ? t.ok : t.codeText, border: `1px solid ${t.copyBorder}`, borderRadius: 6, padding: "2px 8px", fontSize: 12, cursor: "pointer", marginLeft: 10 }}
        >
          {copiedKey === key ? "Copied" : "Copy"}
        </button>
      </div>
      <pre style={{ margin: 0, padding: "12px 14px", background: t.codeBody, color: t.codeText, fontSize: 13, lineHeight: 1.5, overflowX: "auto", whiteSpace: "pre" }}>
        <code style={{ background: "transparent", color: "inherit", padding: 0 }}>{code}</code>
      </pre>
    </div>
  );

  const choice = (options, value, onChange, disabled) => (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
      {options.map((opt) => {
        const active = value === opt.id;
        const off = disabled && disabled.includes(opt.id);
        return (
          <button
            key={opt.id}
            type="button"
            aria-pressed={active}
            disabled={off}
            title={opt.hint || ""}
            onClick={() => onChange(opt.id)}
            style={{
              padding: "6px 12px",
              borderRadius: 10,
              border: `1px solid ${active ? t.activeBorder : t.chipBorder}`,
              background: active ? t.activeBg : t.chip,
              boxShadow: active ? `0 0 0 1px ${t.activeBorder}` : "none",
              color: active ? t.activeText : t.text,
              fontWeight: active ? 600 : 400,
              fontSize: 14,
              cursor: off ? "not-allowed" : "pointer",
              opacity: off ? 0.35 : 1,
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );

  const field = (key, label, value, onChange, hint, placeholder) => (
    <label key={key} style={{ display: "block", fontSize: 13 }}>
      <span style={{ display: "block", marginBottom: 3, color: t.text, fontWeight: 500 }}>{label}</span>
      <input
        type="text"
        className="cgs-input"
        value={value}
        placeholder={placeholder || ""}
        onChange={(e) => onChange(e.target.value)}
        spellCheck={false}
        autoComplete="off"
        style={{ width: "100%", boxSizing: "border-box", padding: "6px 9px", borderRadius: 8, border: `1px solid ${t.fieldBorder}`, background: t.field, color: t.text, fontSize: 14, fontFamily: "inherit" }}
      />
      {hint ? <span style={{ display: "block", marginTop: 2, fontSize: 12, color: t.muted }}>{hint}</span> : null}
    </label>
  );

  const group = (title, children) => (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 6, color: t.text }}>{title}</div>
      {children}
    </div>
  );

  const grid = (children) => (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: 12, marginTop: 8, alignItems: "start" }}>{children}</div>
  );

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  const c = resolve(s);
  const o = s.o || {};
  const set = (patch) => setS((prev) => ({ ...prev, ...patch }));
  const setO = (key, value) => setS((prev) => ({ ...prev, o: { ...(prev.o || {}), [key]: value } }));
  const changeSource = (source) =>
    setS((prev) => {
      const keep = {};
      ["fsName", "fsPath", "proxy", "svcAccount", "ntlmUser"].forEach((k) => {
        if (prev.o && prev.o[k] !== undefined) keep[k] = prev.o[k];
      });
      return { ...prev, source, o: keep };
    });
  const val = (key) => (o[key] !== undefined ? o[key] : "");

  const plan = buildPlan(c);
  const trouble = buildTroubleshooting(c);
  const isWin = s.os === "windows";

  return (
    <div style={{ margin: "20px 0" }}>
      <style>{`.cgs-input::placeholder { color: ${t.placeholder}; opacity: 1; }`}</style>
      <div style={{ border: line, background: t.panel, color: t.text, borderRadius: 16, padding: 20 }}>
        {group("1. Where is the connector installed?", choice(OS_LIST, s.os, (os) => set({ os })))}

        {group(
          "2. What should it reach?",
          <>
            {choice(
              [
                { id: "postgres", label: "PostgreSQL" },
                { id: "mysql", label: "MySQL" },
                { id: "mariadb", label: "MariaDB" },
                { id: "sqlserver", label: "SQL Server" },
                { id: "none", label: "A folder only" },
              ],
              s.source,
              changeSource
            )}
            {s.source !== "none" ? (
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 14, marginTop: 10, color: t.text }}>
                <input type="checkbox" checked={s.withFiles} onChange={(e) => set({ withFiles: e.target.checked })} />
                Also share a folder
              </label>
            ) : null}
          </>
        )}

        {c.isDb
          ? group(
              "3. Connection details",
              <>
                <div style={{ fontSize: 12, color: t.muted }}>Grey values are the defaults. Type to replace one; leave it empty to keep the default.</div>
                {grid(
                <>
                  {field("name", "Connection name", val("name"), (v) => setO("name", v), "The name Timbal sees", c.src.name)}
                  {field("host", "Host", val("host"), (v) => setO("host", v), "As the connector server sees it", c.src.host)}
                  {c.isSqlServer && s.reach === "named_browser"
                    ? field("instance", "Instance name", val("instance"), (v) => setO("instance", v), undefined, "INSTANCE")
                    : field("port", "Port", val("port"), (v) => setO("port", v), undefined, c.port)}
                  {field("db", c.source === "mysql" || c.source === "mariadb" ? "Database (schema)" : "Database", val("db"), (v) => setO("db", v), undefined, c.src.db)}
                  {c.auth === "sql" ? field("user", "Login", val("user"), (v) => setO("user", v), "A read-only login", c.src.user) : null}
                </>
                )}
              </>
            )
          : null}

        {c.isSqlServer ? (
          <>
            {group(
              "How to reach it",
              choice(
                [
                  { id: "default", label: "Default instance, port 1433" },
                  { id: "named_port", label: "Named instance, known port" },
                  { id: "named_browser", label: "Named instance, SQL Browser" },
                ],
                s.reach,
                (reach) => set({ reach })
              )
            )}
            {group(
              "How to log in",
              <>
                {choice(
                  [
                    { id: "sql", label: "SQL login" },
                    { id: "integrated", label: "Windows, as the service", hint: "Windows only" },
                    { id: "windows", label: "Windows, explicit account", hint: "Windows only" },
                  ],
                  c.auth,
                  (auth) => set({ auth }),
                  isWin ? [] : ["integrated", "windows"]
                )}
                {!isWin ? (
                  <div style={{ fontSize: 12, color: t.muted, marginTop: 4 }}>
                    Windows authentication only exists on Windows connectors. On Linux and macOS use a SQL login.
                  </div>
                ) : null}
                {c.auth === "integrated" ? (
                  <div style={{ maxWidth: 380, marginTop: 8 }}>
                    {field("svc", "Service account", val("svcAccount"), (v) => setO("svcAccount", v), "Ends with $ for a group managed service account", "CORP\\svc_timbal$")}
                  </div>
                ) : null}
                {c.auth === "windows" ? (
                  <div style={{ maxWidth: 380, marginTop: 8 }}>
                    {field("ntlm", "Domain account", val("ntlmUser"), (v) => setO("ntlmUser", v), "DOMAIN\\user, typed with a single backslash", "CORP\\svc_timbal")}
                  </div>
                ) : null}
              </>
            )}
            {group(
              "Encryption",
              choice(
                [
                  { id: "trust", label: "Encrypted, self-signed certificate" },
                  { id: "verify", label: "Encrypted and verified" },
                  { id: "none", label: "None" },
                ],
                s.encrypt,
                (encrypt) => set({ encrypt })
              )
            )}
          </>
        ) : null}

        {c.isDb
          ? group(
              "What may Timbal run on this connection?",
              <>
                {choice(
                  [
                    { id: "read", label: "Read queries (default)" },
                    { id: "none", label: "No ad-hoc queries" },
                    { id: "write", label: "Reads and writes" },
                  ],
                  s.query,
                  (query) => set({ query })
                )}
                <div style={{ fontSize: 12, color: t.muted, marginTop: 4 }}>
                  Only for ad-hoc queries. Extracting data and browsing schemas are not affected. This server always has the last word.
                </div>
              </>
            )
          : null}

        {c.isDb
          ? group(
              "Where to keep the connections",
              choice(
                [
                  { id: "file", label: "connections.json" },
                  { id: "inline", label: "Inside config.json (0.1.27+)" },
                ],
                s.layout,
                (layout) => set({ layout })
              )
            )
          : null}

        {c.withFiles
          ? group(
              "Folder to share",
              grid(
                <>
                  {field("fsName", "Share name", val("fsName"), (v) => setO("fsName", v), "The name Timbal sees", "exports")}
                  {field("fsPath", "Path on the server", val("fsPath"), (v) => setO("fsPath", v), "A local folder, or a share already mounted there", FS_PATHS[s.os])}
                </>
              )
            )
          : null}

        {group(
          "Proxy (optional)",
          <div style={{ maxWidth: 420 }}>
            {field("proxy", "Proxy URL", val("proxy"), (v) => setO("proxy", v), "Leave empty if the server connects directly. Needs connector 0.1.33 or newer; the service keeps using it after install", "http://proxy.corp.local:8080")}
          </div>
        )}
      </div>

      <div style={{ marginTop: 18 }}>
        {plan.map((step, idx) => (
          <div key={step.id} style={{ display: "flex", gap: 12, marginBottom: 12, padding: "16px 16px 14px", border: line, borderRadius: 16, background: t.panel, color: t.text }}>
            <div style={{ flex: "0 0 28px", height: 28, borderRadius: 14, background: t.activeBg, border: `1px solid ${t.chipBorder}`, color: t.activeText, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 600 }}>{idx + 1}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
                <div style={{ fontSize: 17, fontWeight: 600, color: t.text }}>{step.title}</div>
                <span style={{ fontSize: 11, border: `1px solid ${t.chipBorder}`, background: t.panel, color: t.muted, borderRadius: 999, padding: "1px 8px", fontWeight: 600 }}>{step.where}</span>
              </div>
              {step.text ? <p style={{ margin: "6px 0 0", lineHeight: 1.55, color: t.text }}>{rich(step.text)}</p> : null}
              {step.blocks.map((b, i) => codeBlock(`${step.id}-${i}`, b.label, b.code))}
              {step.note ? <p style={{ margin: "6px 0 0", fontSize: 13, lineHeight: 1.5, color: t.muted }}>{rich(step.note)}</p> : null}
            </div>
          </div>
        ))}
      </div>

      <details style={{ border: line, background: t.panel, color: t.text, borderRadius: 16, padding: "12px 16px" }}>
        <summary style={{ cursor: "pointer", fontWeight: 600 }}>If something fails, for this setup</summary>
        <ul style={{ margin: "10px 0 0", paddingLeft: 18, lineHeight: 1.55 }}>
          {trouble.map(([problem, fix]) => (
            <li key={problem} style={{ marginBottom: 6 }}>
              <strong>{rich(problem)}</strong>: {rich(fix)}
            </li>
          ))}
        </ul>
        <p style={{ margin: "10px 0 0", fontSize: 13, color: t.muted }}>
          More in <a href="/connect/troubleshooting" style={{ textDecoration: "underline" }}>Troubleshooting</a>.
        </p>
      </details>
    </div>
  );
};
