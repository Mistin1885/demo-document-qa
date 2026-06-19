const PROXY_PREFIX = "/api/proxy";

export function buildBackendTarget(
  backendBase: string,
  requestPathname: string,
  search = ""
): string {
  const base = backendBase.replace(/\/+$/, "");
  let backendPath = requestPathname.startsWith(PROXY_PREFIX)
    ? requestPathname.slice(PROXY_PREFIX.length)
    : requestPathname;

  if (!backendPath) {
    backendPath = "/";
  } else if (!backendPath.startsWith("/")) {
    backendPath = `/${backendPath}`;
  }

  return `${base}${backendPath}${search}`;
}
