How to use
----------

-   ``radiobee aligner`` is a sibling of `bumblebee aligner`. To know more about these aligners, please join qq group `316287378`.

-   Uploaded files should be in pure text format (txt, md, csv etc). ``docx``, ``pdf``, ``srt``, ``html`` etc may be supported later on.
- If ``file 2`` is left blank, ``radibee`` will treat ``file 1`` as mixed English-Chinese text and attempt to separate English and Chinese texts before procedding to align them.

-   Click "Clear" first for subsequent submits when uploading files.
-  ``tf_type`` ``idf_type`` ``dl_type`` ``norm``: Normally there is no need to touch these unless you know what you are doing.
-   Suggested ``esp`` and ``min_samples`` values -- ``esp`` (minimum epsilon): 8-12, ``min_samples``: 4-8.

   -  Larger ``esp`` or smaller ``min_samples`` will result in more aligned pairs but also more **false positives** (pairs falsely identified as candidates). On the other hand, smaller ``esp`` or larger ``min_samples`` values tend to miss 'good' pairs.

-   If you need to have a better look at the image, you can right-click on the image and select copy-image-address and open a new tab in the browser with the copied image address.
-   ``Flag``: Should something go wrong, you can click Flag to save the output and inform the developer.
