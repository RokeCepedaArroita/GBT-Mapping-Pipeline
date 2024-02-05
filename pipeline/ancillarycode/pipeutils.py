def parseSdfitsIndex(infile, mapscans=[]):

    try:
        ifile = open(infile)
    except IOError:
        print("ERROR: Could not open file: {0}\n"
              "Please check and try again.".format(infile))
        raise

    observation = ObservationRows()

    while True:
        line = ifile.readline()
        # look for start of row data or EOF (i.e. not line)
        if '[rows]' in line or not line:
            break

    lookup_table = {}
    header = ifile.readline()

    

    fields = [xx.lstrip() for xx in re.findall(r' *\S+', header)]

    iterator = re.finditer(r' *\S+', header)
    for idx, mm in enumerate(iterator):
        lookup_table[fields[idx]] = slice(mm.start(), mm.end())

    rr = SdFitsIndexRowReader(lookup_table)

    summary = {'WINDOWS': set([]), 'FEEDS': set([])}

    # keep a list of suspect scans so we can know if the
    # user has already been warned
    suspectScans = set()

    for row in ifile:

        rr.setrow(row)

        scanid = int(rr['SCAN'])

        # have a look at the procedure
        #  if it is "Unknown", the data is suspect, so skip it
        procname = rr['PROCEDURE']
        if scanid in suspectScans:
            continue

        if ((scanid not in suspectScans) and procname.lower() == 'unknown'):

            suspectScans.add(scanid)
            if scanid in mapscans:
                print('WARNING: scan', scanid, 'has "Unknown" procedure. Skipping.')
            continue

        feed = int(rr['FDNUM'])
        windowNum = int(rr['IFNUM'])
        pol = int(rr['PLNUM'])
        fitsExtension = int(rr['EXT'])
        rowOfFitsFile = int(rr['ROW'])
        obsid = rr['OBSID']
        procscan = rr['PROCSCAN']
        nchans = rr['NUMCHN']

        summary['WINDOWS'].add((windowNum, float(rr['RESTFREQ'])/1e9))
        summary['FEEDS'].add(rr['FDNUM'])

        # we can assume all integrations of a single scan are within the same
        #   FITS extension
        observation.addRow(scanid, feed, windowNum, pol,
                           fitsExtension, rowOfFitsFile, obsid,
                           procname, procscan, nchans)

    try:
        ifile.close()
    except NameError:
        raise

    return observation, summary
