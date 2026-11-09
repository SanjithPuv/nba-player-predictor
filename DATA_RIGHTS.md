# Data access, sources, and rights

The repository's MIT license covers authored code and synthetic fixtures, not NBA data, reports, trademarks, source websites, or model weights. `nba_api` is a community-maintained client, not an NBA service agreement or permission to redistribute sports data.

Official/primary technical references inspected during implementation:

- [nba_api LeagueGameLog endpoint](https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/leaguegamelog.md): player/team modes P and T, regular-season logs.
- [nba_api project](https://github.com/swar/nba_api): access depends on upstream service behavior; no NBA API key is required by this client, and requests may time out or be denied.
- [NBA official injury report page](https://official.nba.com/nba-injury-report-2026-27-season): timestamped PDFs; historical completeness is not assumed.
- [Example official injury PDF](https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-02-18_06_15AM.pdf): PDF text is archived without inferring entity IDs.
- [NBA schedule JSON](https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json): attempted here, returned HTTP 403. Access from this machine is not assumed.
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) and [license](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE): model card declares Apache-2.0. Verify exact revision and applicable terms before downloading or distributing adapters.
- [NBA terms](https://www.nba.com/termsofuse): review provider terms and your intended use before bulk retrieval, publication or redistribution. This project supplies no data license or legal opinion.

Rate limiting, bounded retries and a successful-response cache reduce unnecessary traffic. Cache identity is season + player/team mode; each refresh writes a new immutable snapshot and updates its manifest. To resume a partial run, repeat ingestion; successful seasons/modes are reused. Use `--refresh` to retrieve changed/current-season data. No scraping or access-control workaround is implemented. A 403 requires access from a permitted environment or an authorized timestamped import.

Do not upload raw datasets, PDF archives, weights, credentials or personal API tokens. The checked-in fixtures use fictional players and explicitly synthetic sources. Reports contain aggregate metrics only, plus a clearly artificial example prediction.
