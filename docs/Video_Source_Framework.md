# Video Source Framework

Dataset Builder resolves every remote or local video through `VideoSourceManager`:

```text
source -> manager -> adapter -> UnifiedVideoAsset -> frame pipeline
```

The frame pipeline receives a verified local path and does not inspect the original hosting site.

## Built-in adapters

- `LocalVideoAdapter`
- `YouTubeAdapter`
- `YandexVideoAdapter`
- `RuTubeAdapter`
- `VimeoAdapter`
- `VKVideoAdapter`
- `OKVideoAdapter`
- `DailymotionAdapter`
- `TwitchVODAdapter`
- `GenericVideoAdapter` (yt-dlp fallback for any HTTP/HTTPS URL)
- `DirectMediaAdapter` (direct HTTP media fallback with resume support)

Remote adapters currently use the existing yt-dlp backend. The adapter contract does not require yt-dlp, so a source may implement its own `probe()` and `download()` methods.

## Adding an adapter

Create a module in `apps/dataset_builder/video_sources/` and define a concrete `VideoSourceAdapter` subclass. Modules are discovered automatically, and concrete subclasses register themselves. Dataset Builder does not need to be edited.

An adapter must define:

- `can_handle(source)`
- `probe(source, **options)`
- `download(source, destination, **options)`

It may override `list_streams()` and `metadata()`. `download()` must return `UnifiedVideoAsset` with a verified local file.

Use a high `priority` for specific sites. Set `fallback_adapter = True` only for generic recovery adapters.

## Metadata

Dataset archives retain both blocks:

- `source_media`: media properties retained for compatibility;
- `video_source`: source type, adapter, original URL/path, strategy, selected stream, fallback state and verification state.

Adapter failures are logged with full technical details. The GUI receives a stable user-facing diagnostic and recovery recommendations.