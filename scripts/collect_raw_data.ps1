param(
    [string]$SeedPath = "data/raw/metadata/seed_urls.csv",
    [string]$RawRoot = "data/raw",
    [bool]$DiscoverTaxRegulations = $true
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Net.Http

function Convert-ToSafeFileName {
    param([string]$Value)

    $safe = $Value.ToLowerInvariant()
    $safe = $safe -replace '[^a-z0-9]+', '_'
    $safe = $safe.Trim('_')
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return "document"
    }
    return $safe
}

function Resolve-DocumentExtension {
    param(
        [string]$Url,
        [string]$DocumentType,
        [string]$ContentType
    )

    if ($DocumentType -eq "pdf" -or $ContentType -match "pdf" -or $Url -match "\.pdf(\?|$)") {
        return @{ Extension = "pdf"; Folder = "pdf" }
    }
    if ($DocumentType -eq "text" -or $ContentType -match "text/plain") {
        return @{ Extension = "txt"; Folder = "text" }
    }
    return @{ Extension = "html"; Folder = "html" }
}

function New-SourceRecord {
    param(
        [string]$SourceId,
        [string]$Title,
        [string]$Url,
        [string]$Topic,
        [string]$DocumentType,
        [string]$Notes
    )

    [pscustomobject]@{
        source_id = $SourceId
        title = $Title
        url = $Url
        topic = $Topic
        document_type = $DocumentType
        notes = $Notes
    }
}

function Find-TaxRegulationPdfSources {
    param([System.Net.Http.HttpClient]$Client)

    $taxUrl = "https://www.pittsburghpa.gov/City-Government/Finances-Budget/Taxes/Tax-Forms"
    $baseUri = [Uri]$taxUrl
    $html = $Client.GetStringAsync($taxUrl).GetAwaiter().GetResult()
    $matches = [regex]::Matches($html, '<a[^>]+href=["''](?<href>[^"'']+\.pdf[^"'']*)["''][^>]*>(?<text>.*?)</a>', 'IgnoreCase')
    $results = New-Object System.Collections.Generic.List[object]

    foreach ($match in $matches) {
        $href = $match.Groups["href"].Value
        $rawText = $match.Groups["text"].Value -replace '<[^>]+>', ' '
        $text = ($rawText -replace '\s+', ' ').Trim()

        if ($text -notmatch "Regulations" -and $href -notmatch "regulation") {
            continue
        }

        $absoluteUrl = ([Uri]::new($baseUri, $href)).AbsoluteUri
        $stem = Convert-ToSafeFileName ($text + "_" + [IO.Path]::GetFileNameWithoutExtension(([Uri]$absoluteUrl).AbsolutePath))
        $sourceId = "city_tax_regulation_$stem"
        if ($sourceId.Length -gt 95) {
            $sourceId = $sourceId.Substring(0, 95).TrimEnd('_')
        }

        $results.Add((New-SourceRecord `
            -SourceId $sourceId `
            -Title $(if ($text) { "City Tax Regulation - $text" } else { "City Tax Regulation PDF" }) `
            -Url $absoluteUrl `
            -Topic "budget_tax" `
            -DocumentType "pdf" `
            -Notes "Discovered from City Tax Forms page")) | Out-Null
    }

    return $results
}

foreach ($folder in @("html", "pdf", "text", "metadata")) {
    New-Item -ItemType Directory -Path (Join-Path $RawRoot $folder) -Force | Out-Null
}

$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(60)
$client.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; CMU-RAG-Assignment/1.0; educational data collection)")

$sources = New-Object System.Collections.Generic.List[object]
Import-Csv $SeedPath | ForEach-Object { $sources.Add($_) | Out-Null }

if ($DiscoverTaxRegulations) {
    try {
        $existingUrls = @{}
        foreach ($source in $sources) {
            $existingUrls[$source.url] = $true
        }
        foreach ($source in (Find-TaxRegulationPdfSources -Client $client)) {
            if (-not $existingUrls.ContainsKey($source.url)) {
                $sources.Add($source) | Out-Null
                $existingUrls[$source.url] = $true
            }
        }
    }
    catch {
        Write-Warning "Could not discover tax regulation PDFs: $($_.Exception.Message)"
    }
}

$metadata = New-Object System.Collections.Generic.List[object]
$crawlTime = (Get-Date).ToString("o")

foreach ($source in $sources) {
    $statusCode = ""
    $contentType = ""
    $localPath = ""
    $bytesLength = 0
    $errorMessage = ""

    try {
        Write-Host "Downloading $($source.source_id): $($source.url)"
        $response = $client.GetAsync($source.url).GetAwaiter().GetResult()
        $statusCode = [int]$response.StatusCode

        if ($response.Content.Headers.ContentType) {
            $contentType = $response.Content.Headers.ContentType.MediaType
        }

        $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        $bytesLength = $bytes.Length

        if ($response.IsSuccessStatusCode) {
            $resolved = Resolve-DocumentExtension -Url $source.url -DocumentType $source.document_type -ContentType $contentType
            $fileName = "$(Convert-ToSafeFileName $source.source_id).$($resolved.Extension)"
            $outputPath = Join-Path (Join-Path $RawRoot $resolved.Folder) $fileName
            [IO.File]::WriteAllBytes((Resolve-Path (Split-Path $outputPath -Parent)).Path + [IO.Path]::DirectorySeparatorChar + (Split-Path $outputPath -Leaf), $bytes)
            $localPath = $outputPath
        }
        else {
            $errorMessage = "HTTP $statusCode"
        }
    }
    catch {
        $errorMessage = $_.Exception.Message
    }

    $metadata.Add([pscustomobject]@{
        source_id = $source.source_id
        title = $source.title
        url = $source.url
        topic = $source.topic
        document_type = $source.document_type
        local_path = $localPath
        crawled_at = $crawlTime
        status_code = $statusCode
        content_type = $contentType
        bytes = $bytesLength
        error = $errorMessage
        notes = $source.notes
    }) | Out-Null
}

$metadataPath = Join-Path $RawRoot "metadata/metadata.csv"
$metadata | Export-Csv -Path $metadataPath -NoTypeInformation -Encoding UTF8

$summaryPath = Join-Path $RawRoot "metadata/download_summary.txt"
$successful = ($metadata | Where-Object { $_.local_path -ne "" }).Count
$failed = $metadata.Count - $successful
$topics = $metadata | Where-Object { $_.local_path -ne "" } | Group-Object topic | Sort-Object Name

$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("Raw data collection summary") | Out-Null
$summary.Add("Crawled at: $crawlTime") | Out-Null
$summary.Add("Total sources attempted: $($metadata.Count)") | Out-Null
$summary.Add("Successful downloads: $successful") | Out-Null
$summary.Add("Failed downloads: $failed") | Out-Null
$summary.Add("") | Out-Null
$summary.Add("Successful downloads by topic:") | Out-Null
foreach ($topic in $topics) {
    $summary.Add("- $($topic.Name): $($topic.Count)") | Out-Null
}
$summary | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "Metadata written to $metadataPath"
Write-Host "Summary written to $summaryPath"
