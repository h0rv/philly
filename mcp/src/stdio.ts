import {
  serveStdio,
  StdioServerTransport,
} from "@modelcontextprotocol/server/stdio";
import { createServer } from "./server.ts";
import { LIMITS } from "./bounds.ts";
// Local trusted process only; the HTTP perimeter is in worker.ts.
serveStdio(() => createServer(), {
  maxSubscriptions: 0,
  transport: new StdioServerTransport(process.stdin, process.stdout, {
    maxBufferSize: LIMITS.requestBytes,
  }),
});
